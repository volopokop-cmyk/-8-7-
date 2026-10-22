"""Telegram-бот клонирования голоса + генерация картинок и 5-секундных GIF.

TTS-модели на выбор (BOT_MODEL или /model):
  qwen-1.7b  - Qwen3-TTS 1.7B, лучшее качество, тяжёлая (~8 ГБ RAM, медленная)
  qwen-0.6b  - Qwen3-TTS 0.6B, хорошее качество, втрое легче
  xtts       - Coqui XTTS-v2, самая лёгкая и быстрая

Картинки/GIF: stabilityai/sd-turbo (лёгкий, CPU/CUDA, ~2.5 ГБ).
  /image <промпт>   — одна картинка
  /gif   <промпт>   — GIF 5 сек из нескольких кадров
                      Ключевые кадры через «|»:
                      /gif кот сидит|кот прыгает|кот приземлился
                      Без «|» — 4 разных сида, кроссфейд до GIF_FRAMES.

Два голосовых режима:
  * Быстрый образец: голосовое/аудио (5-15 сек) -> текст -> озвучка.
  * Свой голос: кнопка «Создать свой голос», записи от 60 сек ->
    личный профиль, использует только владелец.

Качество: ICL (референс + расшифровка Whisper). Для профиля эмбеддинг
усредняется по длинной записи, ICL-якорем служит чистый фрагмент.

Зависимости: pip install python-telegram-bot faster-whisper qwen-tts TTS \
            librosa soundfile safetensors diffusers Pillow
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
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voice-bot")

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}
BOT_MODEL = os.getenv("BOT_MODEL", "qwen-0.6b")
ADMIN_ID = int(os.getenv("ADMIN_ID") or 0)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
MIN_PROFILE_SEC = 60
SR = 24000
QUICK_SEC = 15
ANCHOR_SEC = 14

# ---- картинки ----
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL", "stabilityai/sd-turbo")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))       # должно быть кратно 8
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "2"))       # sd-turbo: 1-4
GIF_FRAMES = int(os.getenv("GIF_FRAMES", "10"))        # кадров в GIF
GIF_SECONDS = 5                                        # длительность GIF
GIF_KEYS = 4                                           # ключевых кадров при одном промпте
PROMPT_MAX = 1000

QWEN_LANGS = {"ru": "Russian", "en": "English", "zh-cn": "Chinese", "ja": "Japanese",
              "ko": "Korean", "de": "German", "fr": "French", "pt": "Portuguese",
              "es": "Spanish", "it": "Italian"}
LANGS = list(QWEN_LANGS)

VOICES = Path("voices")
PENDING = Path("pending")
for d in (VOICES, PENDING):
    d.mkdir(exist_ok=True)

B_CREATE = "🎙 Создать свой голос"
B_MINE = "🧬 Мой голос"
B_DELETE = "🗑 Удалить мой голос"
B_HELP = "ℹ️ Помощь"
B_CANCEL = "❌ Отмена"
B_IMAGE = "🖼 Картинка"
B_GIF = "🎞 Гифка"

B = None                        # текущая TTS-модель
IMG = None                      # картинная модель (ленивая)
tts_lock = asyncio.Lock()       # один тяжёлый расчёт за раз
waiting = 0
QUICK: dict[int, object] = {}
PROFILES: dict[int, object] = {}


# ---------- утилиты ----------
def keyboard(creating: bool = False) -> ReplyKeyboardMarkup:
    if creating:
        rows = [[B_CANCEL]]
    else:
        rows = [[B_CREATE, B_MINE], [B_IMAGE, B_GIF], [B_DELETE, B_HELP]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def profile_path(uid: int) -> Path:
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


# ---------- TTS-модели ----------
class WrongModel(Exception):
    pass


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
        if not text:
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
    """Меняет TTS-модель. Картинную тоже выгружаем, чтобы не было OOM."""
    global B, IMG
    old = B
    old.unload()
    if IMG is not None:
        IMG.unload()
        IMG = None
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


# ---------- картинки ----------
class ImageBackend:
    """Лёгкий text-to-image на sd-turbo (или любом AutoPipelineForText2Image)."""
    name = "sd-turbo"

    def __init__(self, model_id: str = IMAGE_MODEL_ID):
        self.model_id = model_id
        self.pipe = None
        self.device = "cpu"

    def load(self) -> None:
        from diffusers import AutoPipelineForText2Image
        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        dtype = torch.float16 if cuda else torch.float32
        log.info("Loading image model %s on %s (%s)...", self.model_id, self.device, dtype)
        kwargs = {"torch_dtype": dtype}
        if cuda:
            kwargs["variant"] = "fp16"
        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **kwargs).to(self.device)
        except Exception:
            kwargs.pop("variant", None)
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **kwargs).to(self.device)
        log.info("Image model ready.")

    def unload(self) -> None:
        self.pipe = None
        free_memory()

    def generate(self, prompt: str, seed: int | None = None,
                 steps: int = IMAGE_STEPS, size: int = IMAGE_SIZE) -> Image.Image:
        if seed is None:
            seed = int(time.time_ns() % (2**31))
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=g,
            height=size, width=size,
        )
        return out.images[0]


def get_image_backend() -> ImageBackend:
    global IMG
    if IMG is None:
        IMG = ImageBackend()
        IMG.load()
    return IMG


def generate_image_file(prompt: str, out_path: Path) -> None:
    backend = get_image_backend()
    img = backend.generate(prompt)
    img.save(out_path)


def generate_gif_file(prompt: str, out_path: Path) -> None:
    """5-секундный GIF. Ключевые кадры через «|», иначе 4 сида + кроссфейд."""
    backend = get_image_backend()
    base_seed = int(time.time_ns() % (2**31))

    if "|" in prompt:
        keys = [p.strip() for p in prompt.split("|") if p.strip()]
        if not keys:
            raise ValueError("пустые ключевые кадры")
        key_imgs = [backend.generate(k, seed=base_seed + i * 1000) for i, k in enumerate(keys)]
    else:
        n = min(GIF_KEYS, GIF_FRAMES)
        key_imgs = [backend.generate(prompt, seed=base_seed + i * 1000) for i in range(n)]

    n_keys = len(key_imgs)
    frames: list[Image.Image] = []
    for i in range(GIF_FRAMES):
        if n_keys == 1:
            frames.append(key_imgs[0])
            continue
        t = i / max(1, GIF_FRAMES - 1) * (n_keys - 1)
        lo = int(t)
        hi = min(lo + 1, n_keys - 1)
        frac = t - lo
        if frac < 1e-6 or lo == hi:
            frames.append(key_imgs[lo])
        else:
            frames.append(Image.blend(key_imgs[lo], key_imgs[hi], frac))

    duration = max(20, int(GIF_SECONDS * 1000 / GIF_FRAMES))
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)


# ---------- команды и кнопки ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        f"Привет! Я клонирую голос и рисую картинки/GIF. TTS-модель: {B.name}.\n\n"
        "⚡ Быстрый способ: пришли голосовое или аудио (5-15 сек, чистая речь) - потом текст, "
        "и я озвучу его этим голосом.\n"
        "🎙 Свой голос: нажми «Создать свой голос», запиши минимум 1 минуту речи - "
        "я сделаю личный голосовой профиль, доступный только тебе.\n"
        "🖼 Картинка: нажми «🖼 Картинка» и опиши, что нарисовать.\n"
        "🎞 Гифка (5 сек): нажми «🎞 Гифка» и опиши сцену. Для связной анимации задай "
        "ключевые кадры через «|»: «кот сидит|кот прыгает|кот приземлился».\n"
        "/lang ru - язык озвучки (ru, en, zh-cn, ja, ko, de, fr, pt, es, it)\n"
        "/model - какая TTS-модель сейчас работает\n\n"
        "Озвучка идёт на процессоре и может занять от секунд до нескольких минут; "
        "картинки на CPU - 10-40 сек, GIF - до пары минут.\n\n"
        "Используй только голоса, на клонирование которых есть согласие их владельцев. "
        "Запрещено выдавать сгенерированную речь за реального человека, "
        "обманывать и мошенничать.\n\n"
        "Чтобы продолжить, отправь /agree.",
        reply_markup=keyboard(ctx.user_data.get("creating", False)))


async def agree(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    ctx.user_data["agreed"] = True
    await update.message.reply_text(
        "Принято. Пришли образец голоса, нажми «Создать свой голос» или выбери картинку/GIF.",
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
    lines = [f"Сейчас работает TTS: {B.name}", ""] + [f"{k} - {v[0]}" for k, v in MODELS.items()]
    lines += ["", f"Картинки: {IMAGE_MODEL_ID} ({IMAGE_SIZE}x{IMAGE_SIZE}, {IMAGE_STEPS} шагов)",
              f"GIF: {GIF_SECONDS} сек, {GIF_FRAMES} кадров (ключей при одном промпте: {GIF_KEYS})"]
    if not ctx.args:
        if ADMIN_ID and update.effective_user.id == ADMIN_ID:
            lines += ["", "Сменить TTS: /model xtts"]
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
    ctx.user_data.pop("gen_mode", None)
    await update.message.reply_text(
        "Запиши голосовое (или пришли аудио) со своей речью - минимум 1 минута суммарно. "
        "Можно несколькими сообщениями. Говори чётко, без музыки и фона, лучше читать вслух любой текст.\n"
        "Записи используются только для создания профиля и после этого удаляются.",
        reply_markup=keyboard(creating=True))


async def cancel_create(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    shutil.rmtree(PENDING / str(update.effective_user.id), ignore_errors=True)
    ctx.user_data["creating"] = False
    ctx.user_data.pop("gen_mode", None)
    await update.message.reply_text("Создание голоса отменено.", reply_markup=keyboard())


async def use_mine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if has_profile(update.effective_user.id):
        ctx.user_data["mode"] = "profile"
        ctx.user_data.pop("gen_mode", None)
        await update.message.reply_text("Озвучиваю твоим личным голосом. Пришли текст.")
    else:
        await update.message.reply_text("Своего голоса ещё нет - нажми «Создать свой голос».")


async def delete_mine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    for p in VOICES.glob(f"{uid}.*.safetensors"):
        p.unlink(missing_ok=True)
    PROFILES.pop(uid, None)
    QUICK.pop(uid, None)
    shutil.rmtree(PENDING / str(uid), ignore_errors=True)
    ctx.user_data.pop("mode", None)
    ctx.user_data.pop("gen_mode", None)
    ctx.user_data["creating"] = False
    await update.message.reply_text("Твой голос и образцы удалены.", reply_markup=keyboard())


# ---------- картинки: старт/генерация ----------
async def start_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update) or await need_agree(update, ctx):
        return
    if ctx.user_data.get("creating"):
        await update.message.reply_text("Сначала заверши или отмени создание голоса (/cancel).")
        return
    ctx.user_data["gen_mode"] = "image"
    await update.message.reply_text(
        "Опиши картинку. Например: «рыжий кот на подоконнике, утренний свет, кинематографично».\n"
        "Пришли текст - сгенерирую изображение.")


async def start_gif(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update) or await need_agree(update, ctx):
        return
    if ctx.user_data.get("creating"):
        await update.message.reply_text("Сначала заверши или отмени создание голоса (/cancel).")
        return
    ctx.user_data["gen_mode"] = "gif"
    await update.message.reply_text(
        "Опиши GIF длиной 5 секунд. Для связной анимации задай ключевые кадры через «|»:\n"
        "«кот сидит|кот прыгает|кот приземлился»\n"
        "Без «|» сделаю кроссфейд из 4 разных вариантов промпта.\n"
        "Пришли текст - сгенерирую GIF.")


def _check_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("empty")
    if len(prompt) > PROMPT_MAX:
        raise ValueError("long")
    return prompt


async def do_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE, prompt: str):
    if not allowed(update) or await need_agree(update, ctx):
        return
    try:
        prompt = _check_prompt(prompt)
    except ValueError as e:
        if str(e) == "long":
            await update.message.reply_text(f"Слишком длинный запрос (макс {PROMPT_MAX}).")
        else:
            await update.message.reply_text("Пустой запрос.")
        return
    await update.message.reply_text("Генерирую картинку...")
    async with tts_lock:
        try:
            await ctx.bot.send_chat_action(update.message.chat_id, ChatAction.UPLOAD_PHOTO)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "img.png"
            try:
                await asyncio.to_thread(generate_image_file, prompt, out)
            except Exception:
                log.exception("image gen failed")
                await update.message.reply_text(
                    "Не удалось сгенерировать картинку. Проверь, что установлены diffusers и Pillow.")
                return
            with out.open("rb") as fh:
                await update.message.reply_photo(fh, caption=prompt[:900])


async def do_gif(update: Update, ctx: ContextTypes.DEFAULT_TYPE, prompt: str):
    if not allowed(update) or await need_agree(update, ctx):
        return
    try:
        prompt = _check_prompt(prompt)
    except ValueError as e:
        if str(e) == "long":
            await update.message.reply_text(f"Слишком длинный запрос (макс {PROMPT_MAX}).")
        else:
            await update.message.reply_text("Пустой запрос.")
        return
    await update.message.reply_text(f"Генерирую GIF ({GIF_SECONDS} сек), это может занять пару минут...")
    async with tts_lock:
        try:
            await ctx.bot.send_chat_action(update.message.chat_id, ChatAction.UPLOAD_DOCUMENT)
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "anim.gif"
            try:
                await asyncio.to_thread(generate_gif_file, prompt, out)
            except Exception:
                log.exception("gif gen failed")
                await update.message.reply_text(
                    "Не удалось сгенерировать GIF. Проверь diffusers/Pillow и попробуй короче.")
                return
            with out.open("rb") as fh:
                await update.message.reply_animation(fh, caption=prompt[:900])


async def cmd_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await start_image(update, ctx)
    return await do_image(update, ctx, " ".join(ctx.args))


async def cmd_gif(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await start_gif(update, ctx)
    return await do_gif(update, ctx, " ".join(ctx.args))


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
    # аудио перебивает режим картинок/GIF
    ctx.user_data.pop("gen_mode", None)
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
    ctx.user_data.pop("gen_mode", None)
    await msg.reply_text("Голос восстановлен. Пришли текст.")


# ---------- озвучка / картинки по тексту ----------
async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global waiting
    if not allowed(update):
        return
    msg = update.message
    t = msg.text.strip()

    # Кнопки-навигация
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
    if t == B_IMAGE:
        return await start_image(update, ctx)
    if t == B_GIF:
        return await start_gif(update, ctx)

    # Картинки/GIF по нажатой кнопке
    gm = ctx.user_data.get("gen_mode")
    if gm == "image":
        ctx.user_data.pop("gen_mode", None)
        return await do_image(update, ctx, t)
    if gm == "gif":
        ctx.user_data.pop("gen_mode", None)
        return await do_gif(update, ctx, t)

    if ctx.user_data.get("creating"):
        await msg.reply_text("Сейчас идёт запись голоса: пришли аудио или нажми «Отмена».")
        return

    # TTS
    uid = update.effective_user.id
    mode = ctx.user_data.get("mode") or ("profile" if has_profile(uid) else "quick")
    if mode == "profile" and not has_profile(uid):
        mode = "quick"
    if mode == "quick" and uid not in QUICK:
        await msg.reply_text(
            "Сначала пришли голосовое с образцом голоса, нажми «Создать свой голос» "
            "или выбери «🖼 Картинка»/«🎞 Гифка».")
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
    log.info("TTS model %s loaded", B.name)

    app = (Application.builder().token(TOKEN).concurrent_updates(True)
           .read_timeout(60).write_timeout(120).connect_timeout(30).build())
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("agree", agree))
    app.add_handler(CommandHandler("lang", set_lang))
    app.add_handler(CommandHandler("cancel", cancel_create))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("gif", cmd_gif))
    app.add_handler(MessageHandler(filters.Document.FileExtension("qvoice"), on_profile_file))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, on_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
