"""Telegram-бот клонирования голоса на базе Coqui XTTS-v2.

Два режима:
  * Быстрый образец: пришли голосовое/аудио (до 30 сек) -> пришли текст -> озвучка.
  * Свой голос: кнопка «Создать свой голос», записи суммарно от 60 сек ->
    личный голосовой профиль, который использует только его владелец.
"""
import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file
from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    level=logging.INFO)
# Не пишем в публичный лог тексты пользователей и спам getUpdates
logging.getLogger("TTS").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voice-bot")

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}
MIN_PROFILE_SEC = 60     # минимум записи для своего голоса
LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs",
         "ar", "zh-cn", "ja", "hu", "ko", "hi"}

REFS = Path("refs")          # быстрые образцы
VOICES = Path("voices")      # личные голосовые профили
PENDING = Path("pending")    # записи, которые ещё собираются в профиль
for d in (REFS, VOICES, PENDING):
    d.mkdir(exist_ok=True)

B_CREATE = "🎙 Создать свой голос"
B_MINE = "🧬 Мой голос"
B_DELETE = "🗑 Удалить мой голос"
B_HELP = "ℹ️ Помощь"
B_CANCEL = "❌ Отмена"

tts = None                      # модель, грузится при старте
tts_lock = asyncio.Lock()       # CPU один - генерации идут по очереди
waiting = 0                     # сколько запросов сейчас в очереди/в работе


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


# ---------- работа с моделью (блокирующие функции) ----------
def synth_quick(text: str, ref: Path, lang: str, out_wav: Path) -> None:
    tts.tts_to_file(text=text, speaker_wav=str(ref), language=lang,
                    file_path=str(out_wav), split_sentences=True)


def build_profile(wav_paths: list[Path], uid: int, tmp: Path) -> None:
    """Строит личный голосовой профиль (слепок голоса) по длинной записи."""
    model = tts.synthesizer.tts_model
    audio = np.concatenate([sf.read(str(p), dtype="float32")[0] for p in wav_paths])
    combo = tmp / "all.wav"
    sf.write(str(combo), audio, 22050)
    gpt, spk = model.get_conditioning_latents(
        audio_path=[str(combo)], gpt_cond_len=30, gpt_cond_chunk_len=4,
        max_ref_length=60)
    save_file({"gpt": gpt.detach().cpu().clone().contiguous(),
               "spk": spk.detach().cpu().clone().contiguous()},
              str(profile_path(uid)))


def load_profile(path: Path) -> dict:
    d = load_file(str(path))
    gpt, spk = d["gpt"], d["spk"]
    if gpt.ndim != 3 or spk.ndim != 3 or gpt.shape[0] != 1 or spk.shape[0] != 1 \
            or gpt.numel() > 200_000 or spk.numel() > 4096:
        raise ValueError("bad profile")
    return {"gpt": gpt.float(), "spk": spk.float()}


def synth_profile(text: str, uid: int, lang: str, out_wav: Path) -> None:
    model = tts.synthesizer.tts_model
    dev = next(model.parameters()).device
    prof = load_profile(profile_path(uid))
    out = model.inference(text, lang, prof["gpt"].to(dev), prof["spk"].to(dev),
                          enable_text_splitting=True)
    sf.write(str(out_wav), np.asarray(out["wav"], dtype=np.float32), 24000)


# ---------- команды и кнопки ----------
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        "Привет! Я клонирую голос.\n\n"
        "⚡ Быстрый способ: пришли голосовое или аудио (5-30 сек) - потом текст, и я озвучу его этим голосом.\n"
        "🎙 Свой голос: нажми «Создать свой голос», запиши минимум 1 минуту речи - "
        "я сделаю личный голосовой профиль, доступный только тебе.\n"
        "/lang ru - язык озвучки (ru, en, de, fr, es, it, pt, pl, tr, nl, cs, ar, zh-cn, ja, hu, ko, hi)\n\n"
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
        await update.message.reply_text("Пример: /lang ru\nДоступно: " + ", ".join(sorted(LANGS)))
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
    (REFS / f"{uid}.wav").unlink(missing_ok=True)
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
            args = ("-i", str(src), "-t", "300", "-ar", "22050", "-ac", "1", str(dst))
        else:
            dst = REFS / f"{uid}.wav"
            args = ("-i", str(src), "-t", "30", "-ar", "22050", "-ac", "1", str(dst))
        try:
            await asyncio.to_thread(ffmpeg, *args)
        except subprocess.CalledProcessError:
            await msg.reply_text("Не получилось прочитать аудио, попробуй другой файл.")
            return
    if not creating:
        ctx.user_data["mode"] = "quick"
        await msg.reply_text(
            "Быстрый образец сохранён. Пришли текст.\n"
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
        await msg.reply_text("Записей достаточно. Создаю твой голос, это займёт до минуты...")
        async with tts_lock:
            with tempfile.TemporaryDirectory() as tmp:
                await asyncio.to_thread(build_profile, pieces, uid, Path(tmp))
        shutil.rmtree(d, ignore_errors=True)
        ctx.user_data["creating"] = False
        ctx.user_data["mode"] = "profile"
        await msg.reply_text(
            "Готово! Теперь я озвучиваю твоим голосом. Пришли текст.",
            reply_markup=keyboard())
        with profile_path(uid).open("rb") as fh:
            await msg.reply_document(
                fh, filename="my_voice.xtts",
                caption="Резервная копия твоего голоса. Храни её в тайне: по ней можно "
                        "озвучивать твоим голосом. Бот может перезапуститься и забыть профиль - "
                        "просто пришли этот файл мне, и голос восстановится.")
    except Exception:
        log.exception("profile build failed")
        await msg.reply_text("Ошибка при создании голоса, попробуй ещё раз.")
    finally:
        ctx.user_data["building"] = False


async def on_profile_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Восстановление голоса из резервной копии .xtts."""
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
            load_profile(src)
        except Exception:
            await msg.reply_text("Файл голоса повреждён или не подходит.")
            return
        shutil.copy(src, profile_path(uid))
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
    ref = REFS / f"{uid}.wav"
    if mode == "quick" and not ref.exists():
        await msg.reply_text(
            "Сначала пришли голосовое с образцом голоса или нажми «Создать свой голос».")
        return

    lang = ctx.user_data.get("lang", "ru")
    waiting += 1
    try:
        await msg.reply_text(f"Генерирую (в очереди: {waiting})...")
        async with tts_lock:
            await ctx.bot.send_chat_action(msg.chat_id, ChatAction.RECORD_VOICE)
            with tempfile.TemporaryDirectory() as tmp:
                wav, ogg = Path(tmp) / "out.wav", Path(tmp) / "out.ogg"
                try:
                    if mode == "profile":
                        await asyncio.to_thread(synth_profile, t, uid, lang, wav)
                    else:
                        await asyncio.to_thread(synth_quick, t, ref, lang, wav)
                    await asyncio.to_thread(ffmpeg, "-i", str(wav), "-c:a", "libopus",
                                            "-b:a", "48k", str(ogg))
                except Exception:
                    log.exception("synthesis failed")
                    await msg.reply_text("Ошибка генерации, попробуй ещё раз.")
                    return
                with ogg.open("rb") as fh:
                    await msg.reply_voice(fh)
    finally:
        waiting -= 1


def main():
    global tts
    from TTS.api import TTS
    log.info("Loading XTTS-v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
        "cuda" if torch.cuda.is_available() else "cpu")
    log.info("Model loaded")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("agree", agree))
    app.add_handler(CommandHandler("lang", set_lang))
    app.add_handler(CommandHandler("cancel", cancel_create))
    app.add_handler(MessageHandler(filters.Document.FileExtension("xtts"), on_profile_file))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, on_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
