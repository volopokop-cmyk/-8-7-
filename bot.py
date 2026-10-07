"""Telegram-бот клонирования голоса на базе Qwen3-TTS (Base, Apache 2.0).

Два режима:
  * Быстрый образец: пришли голосовое/аудио (5-15 сек) -> пришли текст -> озвучка.
  * Свой голос: кнопка «Создать свой голос», записи суммарно от 60 сек ->
    личный голосовой профиль, который использует только его владелец.

Качество: режим ICL (референс + его расшифровка Whisper). Для профиля эмбеддинг
голоса усредняется по всей длинной записи, а ICL-якорем служит чистый фрагмент.
"""
import asyncio
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
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
MIN_PROFILE_SEC = 60     # минимум записи для своего голоса
SR = 24000               # частота, с которой работает Qwen3-TTS
QUICK_SEC = 15           # длина быстрого образца (лишнее обрезается)
ANCHOR_SEC = 14          # максимум для ICL-фрагмента: длиннее - слишком медленно на CPU

LANGS = {"ru": "Russian", "en": "English", "zh-cn": "Chinese", "ja": "Japanese",
         "ko": "Korean", "de": "German", "fr": "French", "pt": "Portuguese",
         "es": "Spanish", "it": "Italian"}

VOICES = Path("voices")      # личные голосовые профили
PENDING = Path("pending")    # записи, которые ещё собираются в профиль
for d in (VOICES, PENDING):
    d.mkdir(exist_ok=True)

B_CREATE = "🎙 Создать свой голос"
B_MINE = "🧬 Мой голос"
B_DELETE = "🗑 Удалить мой голос"
B_HELP = "ℹ️ Помощь"
B_CANCEL = "❌ Отмена"

tts = None                      # Qwen3TTSModel
whisper = None                  # faster-whisper
tts_lock = asyncio.Lock()       # CPU один - тяжёлые операции идут по очереди
waiting = 0
QUICK: dict[int, object] = {}   # быстрые образцы (только в памяти)
PROFILES: dict[int, object] = {}


# ---------- утилиты ----------
def keyboard(creating: bool = False) -> ReplyKeyboardMarkup:
    rows = [[B_CANCEL]] if creating else [[B_CREATE, B_MINE], [B_DELETE, B_HELP]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def profile_path(uid: int) -> Path:
    return VOICES / f"{uid}.safetensors"


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


# ---------- работа с моделями (блокирующие функции) ----------
def transcribe(audio: np.ndarray, sr: int) -> str:
    a16 = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    segments, _ = whisper.transcribe(a16, beam_size=5, vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()


def make_item(chunks: list[np.ndarray], sr: int = SR):
    """Голосовой профиль (ICL): якорь + расшифровка + средний эмбеддинг всех фрагментов."""
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    if not chunks:
        raise ValueError("no speech")
    step = max(1, len(chunks) // 10)
    embs = []
    for c in chunks[::step][:10]:
        it = tts.create_voice_clone_prompt(ref_audio=(c.astype(np.float32), sr),
                                           x_vector_only_mode=True)[0]
        embs.append(it.ref_spk_embedding.float())
    mean_emb = torch.stack(embs).mean(0)

    target = 10 * sr   # якорь ближе всего к 10 сек
    anchor = min(chunks, key=lambda c: abs(len(c) - target))
    text = transcribe(anchor, sr)
    dev, dt = tts.model.device, tts.model.dtype
    if not text:   # речь не распознана -> режим только по эмбеддингу
        return VoiceClonePromptItem(ref_code=None, ref_spk_embedding=mean_emb.to(dev, dt),
                                    x_vector_only_mode=True, icl_mode=False, ref_text=None)
    base = tts.create_voice_clone_prompt(ref_audio=(anchor.astype(np.float32), sr),
                                         ref_text=text, x_vector_only_mode=False)[0]
    return VoiceClonePromptItem(ref_code=base.ref_code, ref_spk_embedding=mean_emb.to(dev, dt),
                                x_vector_only_mode=False, icl_mode=True, ref_text=text)


def save_item(item, path: Path) -> None:
    tensors = {"spk": item.ref_spk_embedding.detach().float().cpu().clone().contiguous()}
    meta = {"mode": "xvec"}
    if item.icl_mode and item.ref_code is not None:
        tensors["code"] = item.ref_code.detach().cpu().clone().contiguous()
        meta = {"mode": "icl", "ref_text": item.ref_text or ""}
    save_file(tensors, str(path), metadata=meta)


def load_item(path: Path):
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        keys = set(f.keys())
        spk = f.get_tensor("spk")
        code = f.get_tensor("code") if "code" in keys else None
    if spk.ndim != 1 or spk.numel() > 8192:
        raise ValueError("bad spk")
    if code is not None:
        if code.ndim != 2 or code.shape[0] > 800 or code.shape[1] > 64 \
                or int(code.min()) < 0 or int(code.max()) >= 8192:
            raise ValueError("bad code")
        text = meta.get("ref_text", "")
        if not text or len(text) > 2000:
            raise ValueError("bad text")
    dev, dt = tts.model.device, tts.model.dtype
    spk = spk.to(dev, dt)
    if code is None:
        return VoiceClonePromptItem(ref_code=None, ref_spk_embedding=spk,
                                    x_vector_only_mode=True, icl_mode=False, ref_text=None)
    return VoiceClonePromptItem(ref_code=code.long().to(dev), ref_spk_embedding=spk,
                                x_vector_only_mode=False, icl_mode=True, ref_text=text)


def get_profile(uid: int):
    if uid not in PROFILES:
        PROFILES[uid] = load_item(profile_path(uid))
    return PROFILES[uid]


def build_profile(wav_paths: list[Path], uid: int) -> None:
    audio = np.concatenate([sf.read(str(p), dtype="float32")[0] for p in wav_paths])
    item = make_item(split_chunks(audio, SR))
    save_item(item, profile_path(uid))
    PROFILES[uid] = item


def make_quick(wav: Path, uid: int) -> None:
    audio, _ = sf.read(str(wav), dtype="float32")
    audio, _ = librosa.effects.trim(audio, top_db=35)
    QUICK[uid] = make_item([audio])


def synthesize(text: str, item, lang: str, out_wav: Path) -> None:
    pieces = []
    sr = SR
    for chunk in split_text(text):
        wavs, sr = tts.generate_voice_clone(text=chunk, language=lang,
                                            voice_clone_prompt=[item])
        pieces.append(np.asarray(wavs[0], dtype=np.float32))
        pieces.append(np.zeros(int(0.25 * sr), dtype=np.float32))
    sf.write(str(out_wav), np.concatenate(pieces[:-1]), sr)


# ---------- команды и кнопки ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        "Привет! Я клонирую голос (Qwen3-TTS).\n\n"
        "⚡ Быстрый способ: пришли голосовое или аудио (5-15 сек, чистая речь) - потом текст, "
        "и я озвучу его этим голосом.\n"
        "🎙 Свой голос: нажми «Создать свой голос», запиши минимум 1 минуту речи - "
        "я сделаю личный голосовой профиль, доступный только тебе. Так качество выше.\n"
        "/lang ru - язык озвучки (ru, en, zh-cn, ja, ko, de, fr, pt, es, it)\n\n"
        "Озвучка идёт на процессоре и занимает от десятков секунд до нескольких минут.\n\n"
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
    profile_path(uid).unlink(missing_ok=True)
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
            load_item(src)
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

    lang = LANGS[ctx.user_data.get("lang", "ru")]
    waiting += 1
    try:
        await msg.reply_text(f"Генерирую (в очереди: {waiting}). На процессоре это может занять несколько минут...")
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
    global tts, whisper
    from faster_whisper import WhisperModel
    from qwen_tts import Qwen3TTSModel
    cuda = torch.cuda.is_available()
    log.info("Loading %s ...", QWEN_MODEL)
    tts = Qwen3TTSModel.from_pretrained(
        QWEN_MODEL,
        device_map="cuda:0" if cuda else "cpu",
        dtype=torch.bfloat16 if cuda else torch.float32)
    log.info("Loading Whisper %s ...", WHISPER_MODEL)
    whisper = WhisperModel(WHISPER_MODEL, device="cuda" if cuda else "cpu",
                           compute_type="float16" if cuda else "int8")
    log.info("Models loaded")

    app = (Application.builder().token(TOKEN).concurrent_updates(True)
           .read_timeout(60).write_timeout(120).connect_timeout(30).build())
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("agree", agree))
    app.add_handler(CommandHandler("lang", set_lang))
    app.add_handler(CommandHandler("cancel", cancel_create))
    app.add_handler(MessageHandler(filters.Document.FileExtension("qvoice"), on_profile_file))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, on_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
