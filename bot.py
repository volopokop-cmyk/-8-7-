"""Telegram-бот клонирования голоса. Модели на выбор (BOT_MODEL или /model):
  qwen-1.7b  - Qwen3-TTS 1.7B, лучшее качество, тяжёлая (~8 ГБ RAM, медленная)
  qwen-0.6b  - Qwen3-TTS 0.6B, хорошее качество, втрое легче
  xtts       - Coqui XTTS-v2, самая лёгкая и быстрая

Два режима:
  * Быстрый образец: пришли голосовое/аудио (5-15 сек) -> пришли текст -> озвучка.
  * Свой голос: кнопка «Создать свой голос», записи суммарно от 60 сек ->
    личный голосовой профиль, который использует только его владелец.

Качество: режим ICL (референс + его расшифровка Whisper). Для профиля эмбеддинг
голоса усредняется по всей длинной записи, а ICL-якорем служит чистый фрагмент.
"""
import asyncio
import gc
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    level=logging.INFO)
# Не пишем в публичный лог тексты пользователей и спам getUpdates
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voice-bot")

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}
BOT_MODEL = os.getenv("BOT_MODEL", "qwen-0.6b")
ADMIN_ID = int(os.getenv("ADMIN_ID") or 0)           # кто может менять модель командой /model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # для Qwen (расшифровка образца); medium точнее, но тяжелее
MIN_PROFILE_SEC = 60     # минимум записи для своего голоса
SR = 24000               # частота, с которой работает Qwen3-TTS
QUICK_SEC = 15           # длина быстрого образца (лишнее обрезается)
ANCHOR_SEC = 14          # максимум для ICL-фрагмента: длиннее - слишком медленно на CPU

QWEN_LANGS = {"ru": "Russian", "en": "English", "zh-cn": "Chinese", "ja": "Japanese",
              "ko": "Korean", "de": "German", "fr": "French", "pt": "Portuguese",
              "es": "Spanish", "it": "Italian"}
LANGS = list(QWEN_LANGS)     # общий набор языков, который понимают все модели

VOICES = Path("voices")      # личные голосовые профили
PENDING = Path("pending")    # записи, которые ещё собираются в профиль
for d in (VOICES, PENDING):
    d.mkdir(exist_ok=True)

B_CREATE = "🎙 Создать свой голос"
B_MINE = "🧬 Мой голос"
B_DELETE = "🗑 Удалить мой голос"
B_HELP = "ℹ️ Помощь"
B_CANCEL = "❌ Отмена"

B = None                        # текущая модель (Backend)
tts_lock = asyncio.Lock()       # CPU один - тяжёлые операции идут по очереди
waiting = 0
QUICK: dict[int, object] = {}   # быстрые образцы (только в памяти, привязаны к модели)
PROFILES: dict[int, object] = {}


# ---------- утилиты ----------
def keyboard(creating: bool = False) -> ReplyKeyboardMarkup:
    rows = [[B_CANCEL]] if creating else [[B_CREATE, B_MINE], [B_DELETE, B_HELP]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def profile_path(uid: int) -> Path:
    # профиль зависит от модели: у каждой свой формат эмбеддингов
    return VOICES / f"{uid}.{B.name}.safetensors"


def has_profile(uid: int) -> bool:
    return profile_path(uid).exists()


def allowed(update: Update) -> bool:
    return not ALLOWED or update.effective_user.id in ALLOWED


async def need_agree(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    if ctx.user_data.get("agreed"):
        return False
    await update.message.reply_text("Сначала прими правила: /agree")
    return True


def split_text(text: str, limit: int = 220) -> list[str]:
    """Режем длинный текст по предложениям: так Qwen озвучивает его стабильнее."""
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    out, cur = [], ""
    for p in (x.strip() for x in parts):
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    final = []
    for c in out:
        while len(c) > limit * 1.5:
            cut = c.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            final.append(c[:cut])
            c = c[cut:].strip()
        final.append(c)
    return final


def split_chunks(audio: np.ndarray, sr: int, max_sec: float = ANCHOR_SEC,
                 min_sec: float = 3.0) -> list[np.ndarray]:
    """Режем запись на фрагменты по паузам, каждый не длиннее max_sec."""
    iv = librosa.effects.split(audio, top_db=35)
    spans, cs, ce = [], None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif (e - cs) / sr <= max_sec:
            ce = e
        else:
            spans.append((cs, ce))
            cs, ce = s, e
    if cs is not None:
        spans.append((cs, ce))
    out = []
    step = int(max_sec * sr)
    for s, e in spans:
        while (e - s) > step * 1.5:
            out.append(audio[s:s + step])
            s += step
        if (e - s) / sr >= min_sec:
            out.append(audio[s:e])
    return out


# ---------- модели ----------
class WrongModel(Exception):
    """Профиль создан для другой модели."""


def _save(tensors: dict, path: Path, meta: dict) -> None:
    save_file({k: v.detach().cpu().clone().contiguous() for k, v in tensors.items()},
              str(path), metadata=meta)


def _read(path: Path, name: str):
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if meta.get("backend") != name:
        raise WrongModel(meta.get("backend", "?"))
    return meta, tensors


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QwenBackend:
    def __init__(self, name: str, model_id: str):
        self.name, self.model_id = name, model_id
        self.tts = self.whisper = None

    def load(self) -> None:
        from faster_whisper import WhisperModel
        from qwen_tts import Qwen3TTSModel
        cuda = torch.cuda.is_available()
        dtype = getattr(torch, os.getenv("QWEN_DTYPE") or ("bfloat16" if cuda else "float32"))
        log.info("Loading %s (%s)...", self.model_id, dtype)
        self.tts = Qwen3TTSModel.from_pretrained(
            self.model_id, device_map="cuda:0" if cuda else "cpu", dtype=dtype)
        log.info("Loading Whisper %s...", WHISPER_MODEL)
        self.whisper = WhisperModel(WHISPER_MODEL, device="cuda" if cuda else "cpu",
                                    compute_type="float16" if cuda else "int8")

    def unload(self) -> None:
        self.tts = self.whisper = None
        free_memory()

    def lang(self, code: str) -> str:
        return QWEN_LANGS[code]

    def _transcribe(self, audio: np.ndarray, sr: int) -> str:
        a16 = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        segments, _ = self.whisper.transcribe(a16, beam_size=5, vad_filter=False)
        return " ".join(x.text.strip() for x in segments).strip()

    def make_item(self, chunks: list[np.ndarray]):
        """ICL: якорь + расшифровка + средний эмбеддинг всех фрагментов."""
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        if not chunks:
            raise ValueError("no speech")
        step = max(1, len(chunks) // 10)
        embs = []
        for c in chunks[::step][:10]:
            it = self.tts.create_voice_clone_prompt(
                ref_audio=(c.astype(np.float32), SR), x_vector_only_mode=True)[0]
            embs.append(it.ref_spk_embedding.float())
        mean_emb = torch.stack(embs).mean(0)
        anchor = min(chunks, key=lambda c: abs(len(c) - 10 * SR))
        text = self._transcribe(anchor, SR)
        dev, dt = self.tts.model.device, self.tts.model.dtype
        if not text:   # речь не распознана -> только по эмбеддингу
            return VoiceClonePromptItem(ref_code=None, ref_spk_embedding=mean_emb.to(dev, dt),
                                        x_vector_only_mode=True, icl_mode=False, ref_text=None)
        base = self.tts.create_voice_clone_prompt(
            ref_audio=(anchor.astype(np.float32), SR), ref_text=text, x_vector_only_mode=False)[0]
        return VoiceClonePromptItem(ref_code=base.ref_code, ref_spk_embedding=mean_emb.to(dev, dt),
                                    x_vector_only_mode=False, icl_mode=True, ref_text=text)

    def save_item(self, item, path: Path) -> None:
        tensors = {"spk": item.ref_spk_embedding.float()}
        meta = {"backend": self.name, "mode": "xvec"}
        if item.icl_mode and item.ref_code is not None:
            tensors["code"] = item.ref_code
            meta.update(mode="icl", ref_text=item.ref_text or "")
        _save(tensors, path, meta)

    def load_item(self, path: Path):
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        meta, t = _read(path, self.name)
        spk, code = t["spk"], t.get("code")
        if spk.ndim != 1 or spk.numel() > 8192:
            raise ValueError("bad spk")
        dev, dt = self.tts.model.device, self.tts.model.dtype
        if code is None:
            return VoiceClonePromptItem(ref_code=None, ref_spk_embedding=spk.to(dev, dt),
                                        x_vector_only_mode=True, icl_mode=False, ref_text=None)
        text = meta.get("ref_text", "")
        if code.ndim != 2 or code.shape[0] > 800 or code.shape[1] > 64 \
                or int(code.min()) < 0 or int(code.max()) >= 8192 or not text or len(text) > 2000:
            raise ValueError("bad profile")
        return VoiceClonePromptItem(ref_code=code.long().to(dev), ref_spk_embedding=spk.to(dev, dt),
                                    x_vector_only_mode=False, icl_mode=True, ref_text=text)

    def synth_chunk(self, text: str, item, lang: str):
        wavs, sr = self.tts.generate_voice_clone(
            text=text, language=self.lang(lang), voice_clone_prompt=[item])
        return np.asarray(wavs[0], dtype=np.float32), sr


class XttsBackend:
    name = "xtts"

    def __init__(self):
        self.tts = None

    def load(self) -> None:
        from TTS.api import TTS
        logging.getLogger("TTS").setLevel(logging.WARNING)
        log.info("Loading XTTS-v2...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            "cuda" if torch.cuda.is_available() else "cpu")

    def unload(self) -> None:
        self.tts = None
        free_memory()

    def make_item(self, chunks: list[np.ndarray]):
        if not chunks:
            raise ValueError("no speech")
        model = self.tts.synthesizer.tts_model
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ref.wav"
            sf.write(str(p), np.concatenate(chunks).astype(np.float32), SR)
            gpt, spk = model.get_conditioning_latents(
                audio_path=[str(p)], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
        return {"gpt": gpt.detach().cpu().clone(), "spk": spk.detach().cpu().clone()}

    def save_item(self, item, path: Path) -> None:
        _save(item, path, {"backend": self.name})

    def load_item(self, path: Path):
        _, t = _read(path, self.name)
        gpt, spk = t["gpt"], t["spk"]
        if gpt.ndim != 3 or spk.ndim != 3 or gpt.shape[0] != 1 or spk.shape[0] != 1 \
                or gpt.numel() > 200_000 or spk.numel() > 4096:
            raise ValueError("bad profile")
        return {"gpt": gpt.float(), "spk": spk.float()}

    def synth_chunk(self, text: str, item, lang: str):
        model = self.tts.synthesizer.tts_model
        dev = next(model.parameters()).device
        out = model.inference(text, lang, item["gpt"].to(dev), item["spk"].to(dev),
                              enable_text_splitting=True)
        return np.asarray(out["wav"], dtype=np.float32), 24000


MODELS = {
    "qwen-1.7b": ("Qwen3-TTS 1.7B: лучшее качество, тяжёлая и медленная",
                  lambda: QwenBackend("qwen-1.7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")),
    "qwen-0.6b": ("Qwen3-TTS 0.6B: хорошее качество, втрое легче",
                  lambda: QwenBackend("qwen-0.6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")),
    "xtts": ("XTTS-v2: самая лёгкая и быстрая",
             lambda: XttsBackend()),
}


def swap_model(name: str) -> None:
    """Меняет модель на лету. Если новая не загрузилась - возвращает прежнюю."""
    global B
    old = B
    old.unload()
    QUICK.clear()
    PROFILES.clear()
    try:
        new = MODELS[name][1]()
        new.load()
        B = new
    except BaseException:
        log.exception("model %s failed to load, restoring %s", name, old.name)
        free_memory()
        old.load()
        B = old
        raise


def get_profile(uid: int):
    if uid not in PROFILES:
        PROFILES[uid] = B.load_item(profile_path(uid))
    return PROFILES[uid]


def build_profile(wav_paths: list[Path], uid: int) -> None:
    audio = np.concatenate([sf.read(str(p), dtype="float32")[0] for p in wav_paths])
    item = B.make_item(split_chunks(audio, SR))
    B.save_item(item, profile_path(uid))
    PROFILES[uid] = item


def make_quick(wav: Path, uid: int) -> None:
    audio, _ = sf.read(str(wav), dtype="float32")
    audio, _ = librosa.effects.trim(audio, top_db=35)
    QUICK[uid] = B.make_item([audio])


def synthesize(text: str, item, lang: str, out_wav: Path) -> None:
    pieces, sr = [], SR
    for chunk in split_text(text):
        wav, sr = B.synth_chunk(chunk, item, lang)
        pieces += [wav, np.zeros(int(0.25 * sr), dtype=np.float32)]
    sf.write(str(out_wav), np.concatenate(pieces[:-1]), sr)


# ---------- команды и кнопки ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        f"Привет! Я клонирую голос. Модель: {B.name}.\n\n"
        "⚡ Быстрый способ: пришли голосовое или аудио (5-15 сек, чистая речь) - потом текст, "
        "и я озвучу его этим голосом.\n"
        "🎙 Свой голос: нажми «Создать свой голос», запиши минимум 1 минуту речи - "
        "я сделаю личный голосовой профиль, доступный только тебе. Так качество выше.\n"
        "/lang ru - язык озвучки (ru, en, zh-cn, ja, ko, de, fr, pt, es, it)\n"
        "/model - какая модель сейчас работает\n\n"
        "Озвучка идёт на процессоре и может занять от секунд до нескольких минут.\n\n"
        "Используй только голоса, на клонирование которых есть согласие их владельцев. "
        "Запрещено выдавать сгенерированную речь за реального человека, "
        "обманывать и мошенничать.\n\n"
        "Чтобы продолжить, отправь /agree - это подтверждение, что ты согласен с этими правилами.",
        reply_markup=keyboard(ctx.user_data.get("creating", False)))


async def agree(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    ctx.user_data["agreed"] = True
    await update.message.reply_text(
        "Принято. Пришли образец голоса или нажми «Создать свой голос».",
        reply_markup=keyboard())


async def set_lang(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    lang = (ctx.args[0].lower() if ctx.args else "")
    if lang not in LANGS:
        await update.message.reply_text("Пример: /lang ru\nДоступно: " + ", ".join(LANGS))
        return
    ctx.user_data["lang"] = lang
    await update.message.reply_text(f"Язык озвучки: {lang}")


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    lines = [f"Сейчас работает: {B.name}", ""] + [f"{k} - {v[0]}" for k, v in MODELS.items()]
    if not ctx.args:
        if ADMIN_ID and update.effective_user.id == ADMIN_ID:
            lines += ["", "Сменить: /model xtts"]
        await update.message.reply_text("\n".join(lines))
        return
    if not ADMIN_ID or update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Менять модель может только администратор.")
        return
    name = ctx.args[0].lower()
    if name not in MODELS:
        await update.message.reply_text("\n".join(lines))
        return
    if name == B.name:
        await update.message.reply_text("Эта модель уже работает.")
        return
    await update.message.reply_text(f"Загружаю {name}, подожди пару минут. Образцы голосов придётся прислать заново.")
    async with tts_lock:
        try:
            await asyncio.to_thread(swap_model, name)
        except BaseException:
            await update.message.reply_text(f"Не удалось загрузить {name}, вернул {B.name}.")
            return
    await update.message.reply_text(f"Готово, теперь работает {B.name}.")


async def begin_create(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    shutil.rmtree(PENDING / str(uid), ignore_errors=True)
    ctx.user_data["creating"] = True
    await update.message.reply_text(
        "Запиши голосовое (или пришли аудио) со своей речью - минимум 1 минута суммарно. "
        "Можно несколькими сообщениями. Говори чётко, без музыки и фона, лучше читать вслух любой текст.\n"
        "Записи используются только для создания профиля и после этого удаляются.",
        reply_markup=keyboard(creating=True))


async def cancel_create(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    shutil.rmtree(PENDING / str(update.effective_user.id), ignore_errors=True)
    ctx.user_data["creating"] = False
    await update.message.reply_text("Создание голоса отменено.", reply_markup=keyboard())


async def use_mine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if has_profile(update.effective_user.id):
        ctx.user_data["mode"] = "profile"
        await update.message.reply_text("Озвучиваю твоим личным голосом. Пришли текст.")
    else:
        await update.message.reply_text("Своего голоса ещё нет - нажми «Создать свой голос».")


async def delete_mine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    for p in VOICES.glob(f"{uid}.*.safetensors"):   # профили всех моделей
        p.unlink(missing_ok=True)
    PROFILES.pop(uid, None)
    QUICK.pop(uid, None)
    shutil.rmtree(PENDING / str(uid), ignore_errors=True)
    ctx.user_data.pop("mode", None)
    ctx.user_data["creating"] = False
    await update.message.reply_text("Твой голос и образцы удалены.", reply_markup=keyboard())


# ---------- приём аудио ----------
async def on_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update) or await need_agree(update, ctx):
        return
    msg = update.message
    media = msg.voice or msg.audio or msg.document
    if media is None:
        return
    uid = update.effective_user.id
    creating = ctx.user_data.get("creating", False)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in"
        try:
            f = await ctx.bot.get_file(media.file_id)
            await f.download_to_drive(src)
        except Exception:
            await msg.reply_text("Не удалось скачать файл (Telegram разрешает ботам до 20 МБ).")
            return
        if creating:
            d = PENDING / str(uid)
            d.mkdir(parents=True, exist_ok=True)
            dst = d / f"{time.time_ns()}.wav"
            args = ("-i", str(src), "-t", "300", "-ar", str(SR), "-ac", "1", str(dst))
        else:
            dst = Path(tmp) / "quick.wav"
            args = ("-i", str(src), "-t", str(QUICK_SEC), "-ar", str(SR), "-ac", "1", str(dst))
        try:
            await asyncio.to_thread(ffmpeg, *args)
        except subprocess.CalledProcessError:
            await msg.reply_text("Не получилось прочитать аудио, попробуй другой файл.")
            return
        if not creating:
            await msg.reply_text("Анализирую образец голоса...")
            try:
                async with tts_lock:
                    await asyncio.to_thread(make_quick, dst, uid)
            except Exception:
                log.exception("quick sample failed")
                await msg.reply_text("Не удалось обработать образец: нужна чистая речь, хотя бы 3 секунды.")
                return
            ctx.user_data["mode"] = "quick"
            await msg.reply_text(
                "Образец принят. Пришли текст.\n"
                "(Чтобы вернуться к своему голосу, нажми «🧬 Мой голос».)")
            return
    await continue_create(update, ctx)


async def continue_create(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    uid = update.effective_user.id
    d = PENDING / str(uid)
    pieces = sorted(d.glob("*.wav"))
    total = sum(sf.info(str(p)).duration for p in pieces)
    if total < MIN_PROFILE_SEC:
        await msg.reply_text(
            f"Записано {int(total)} сек из {MIN_PROFILE_SEC}. "
            f"Пришли ещё минимум {int(MIN_PROFILE_SEC - total) + 1} сек.")
        return
    if ctx.user_data.get("building"):
        return
    ctx.user_data["building"] = True
    try:
        await msg.reply_text("Записей достаточно. Создаю твой голос, это займёт несколько минут...")
        async with tts_lock:
            await asyncio.to_thread(build_profile, pieces, uid)
        shutil.rmtree(d, ignore_errors=True)
        ctx.user_data["creating"] = False
        ctx.user_data["mode"] = "profile"
        await msg.reply_text(
            "Готово! Теперь я озвучиваю твоим голосом. Пришли текст.",
            reply_markup=keyboard())
        with profile_path(uid).open("rb") as fh:
            await msg.reply_document(
                fh, filename="my_voice.qvoice",
                caption="Резервная копия твоего голоса. Храни её в тайне: по ней можно "
                        "озвучивать твоим голосом. Бот может перезапуститься и забыть профиль - "
                        "просто пришли этот файл мне, и голос восстановится.")
    except Exception:
        log.exception("profile build failed")
        await msg.reply_text("Не удалось создать голос. Нужна чистая речь без музыки, попробуй ещё раз.")
    finally:
        ctx.user_data["building"] = False


async def on_profile_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Восстановление голоса из резервной копии .qvoice."""
    if not allowed(update) or await need_agree(update, ctx):
        return
    msg = update.message
    if msg.document.file_size and msg.document.file_size > 1_000_000:
        await msg.reply_text("Это не похоже на файл голоса.")
        return
    uid = update.effective_user.id
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "p.safetensors"
        f = await ctx.bot.get_file(msg.document.file_id)
        await f.download_to_drive(src)
        try:
            await asyncio.to_thread(B.load_item, src)
        except WrongModel as e:
            await msg.reply_text(
                f"Этот голос создан для модели «{e}», а сейчас работает «{B.name}». "
                "Профили разных моделей несовместимы: создай голос заново или попроси администратора переключить модель.")
            return
        except Exception:
            await msg.reply_text("Файл голоса повреждён или не подходит.")
            return
        shutil.copy(src, profile_path(uid))
    PROFILES.pop(uid, None)
    ctx.user_data["mode"] = "profile"
    await msg.reply_text("Голос восстановлен. Пришли текст.")


# ---------- озвучка ----------
async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global waiting
    if not allowed(update):
        return
    msg = update.message
    t = msg.text.strip()
    if t == B_HELP:
        return await start(update, ctx)
    if await need_agree(update, ctx):
        return
    if t == B_CREATE:
        return await begin_create(update, ctx)
    if t == B_CANCEL:
        return await cancel_create(update, ctx)
    if t == B_MINE:
        return await use_mine(update, ctx)
    if t == B_DELETE:
        return await delete_mine(update, ctx)
    if ctx.user_data.get("creating"):
        await msg.reply_text("Сейчас идёт запись голоса: пришли аудио или нажми «Отмена».")
        return

    uid = update.effective_user.id
    mode = ctx.user_data.get("mode") or ("profile" if has_profile(uid) else "quick")
    if mode == "profile" and not has_profile(uid):
        mode = "quick"
    if mode == "quick" and uid not in QUICK:
        await msg.reply_text(
            "Сначала пришли голосовое с образцом голоса или нажми «Создать свой голос».")
        return

    lang = ctx.user_data.get("lang", "ru")
    waiting += 1
    try:
        await msg.reply_text(f"Генерирую (в очереди: {waiting}). Это может занять до нескольких минут...")
        async with tts_lock:
            await ctx.bot.send_chat_action(msg.chat_id, ChatAction.RECORD_VOICE)
            with tempfile.TemporaryDirectory() as tmp:
                wav, ogg = Path(tmp) / "out.wav", Path(tmp) / "out.ogg"
                try:
                    item = await asyncio.to_thread(get_profile, uid) if mode == "profile" else QUICK[uid]
                    await asyncio.to_thread(synthesize, t, item, lang, wav)
                    await asyncio.to_thread(ffmpeg, "-i", str(wav), "-c:a", "libopus",
                                            "-b:a", "64k", str(ogg))
                except Exception:
                    log.exception("synthesis failed")
                    await msg.reply_text("Ошибка генерации, попробуй ещё раз.")
                    return
                with ogg.open("rb") as fh:
                    await msg.reply_voice(fh)
    finally:
        waiting -= 1


def main():
    global B
    if BOT_MODEL not in MODELS:
        raise SystemExit(f"Неизвестная модель {BOT_MODEL}. Доступно: {', '.join(MODELS)}")
    B = MODELS[BOT_MODEL][1]()
    B.load()
    log.info("Model %s loaded", B.name)

    app = (Application.builder().token(TOKEN).concurrent_updates(True)
           .read_timeout(60).write_timeout(120).connect_timeout(30).build())
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("agree", agree))
    app.add_handler(CommandHandler("lang", set_lang))
    app.add_handler(CommandHandler("cancel", cancel_create))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(MessageHandler(filters.Document.FileExtension("qvoice"), on_profile_file))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, on_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
