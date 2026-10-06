"""Telegram-бот клонирования голоса на базе Coqui XTTS-v2.

Использование:
  1. Отправьте боту голосовое сообщение / аудио (5-30 сек) - это образец голоса.
  2. Отправьте текст - бот озвучит его клонированным голосом.
  /lang ru|en|de|... - язык озвучки.
"""
import asyncio
import logging
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    level=logging.INFO)
log = logging.getLogger("voice-bot")

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}
MAX_CHARS = 500
COOLDOWN_SEC = 30        # пауза между генерациями одного пользователя
DAILY_LIMIT = 15         # генераций в сутки на пользователя
MAX_QUEUE = 4            # сколько запросов максимум ждёт очереди
LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs",
         "ar", "zh-cn", "ja", "hu", "ko", "hi"}
REFS = Path("refs")
REFS.mkdir(exist_ok=True)

tts = None                      # модель, грузится при старте
tts_lock = asyncio.Lock()       # CPU слабый - одна генерация за раз
waiting = 0                     # сколько запросов сейчас в очереди/в работе
last_used: dict[int, float] = {}
daily: dict[tuple[int, str], int] = defaultdict(int)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def synthesize(text: str, ref: Path, lang: str, out_wav: Path) -> None:
    tts.tts_to_file(text=text, speaker_wav=str(ref), language=lang,
                    file_path=str(out_wav), split_sentences=True)


def allowed(update: Update) -> bool:
    return not ALLOWED or update.effective_user.id in ALLOWED


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        "Привет! Я клонирую голос.\n\n"
        "1) Пришли голосовое или аудио (5-30 сек, чистая речь) - образец голоса.\n"
        "2) Пришли текст - я озвучу его этим голосом.\n"
        "/lang ru - язык озвучки (ru, en, de, fr, es, it, pt, pl, tr, nl, cs, ar, zh-cn, ja, hu, ko, hi)\n\n"
        "Используй только голоса, на клонирование которых есть согласие их владельцев. "
        "Запрещено выдавать сгенерированную речь за реального человека, "
        "обманывать и мошенничать.\n\n"
        "Чтобы продолжить, отправь /agree - это подтверждение, что ты согласен с этими правилами.")


async def agree(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    ctx.user_data["agreed"] = True
    await update.message.reply_text("Принято. Пришли голосовое сообщение с образцом голоса.")


async def need_agree(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    if ctx.user_data.get("agreed"):
        return False
    await update.message.reply_text("Сначала прими правила: /agree")
    return True


async def set_lang(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    lang = (ctx.args[0].lower() if ctx.args else "")
    if lang not in LANGS:
        await update.message.reply_text("Пример: /lang ru\nДоступно: " + ", ".join(sorted(LANGS)))
        return
    ctx.user_data["lang"] = lang
    await update.message.reply_text(f"Язык озвучки: {lang}")


async def on_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    msg = update.message
    if await need_agree(update, ctx):
        return
    media = msg.voice or msg.audio or msg.document
    if media is None:
        return
    if msg.document and not (msg.document.mime_type or "").startswith("audio"):
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in"
        f = await ctx.bot.get_file(media.file_id)
        await f.download_to_drive(src)
        dst = REFS / f"{update.effective_user.id}.wav"
        try:
            await asyncio.to_thread(ffmpeg, "-i", str(src), "-t", "30",
                                    "-ar", "22050", "-ac", "1", str(dst))
        except subprocess.CalledProcessError:
            await msg.reply_text("Не получилось прочитать аудио, попробуй другой файл.")
            return
    await msg.reply_text("Образец голоса сохранён. Теперь пришли текст.")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    global waiting
    msg = update.message
    if await need_agree(update, ctx):
        return
    text = msg.text.strip()
    uid = update.effective_user.id
    ref = REFS / f"{update.effective_user.id}.wav"
    if not ref.exists():
        await msg.reply_text("Сначала пришли голосовое сообщение с образцом голоса.")
        return
    if len(text) > MAX_CHARS:
        await msg.reply_text(f"Слишком длинно: максимум {MAX_CHARS} символов.")
        return
    now = time.time()
    wait = COOLDOWN_SEC - (now - last_used.get(uid, 0))
    if wait > 0:
        await msg.reply_text(f"Подожди ещё {int(wait) + 1} сек.")
        return
    day_key = (uid, time.strftime("%Y-%m-%d"))
    if daily[day_key] >= DAILY_LIMIT:
        await msg.reply_text(f"Дневной лимит ({DAILY_LIMIT} генераций) исчерпан, приходи завтра.")
        return
    if waiting >= MAX_QUEUE:
        await msg.reply_text("Сейчас много запросов, попробуй через минуту.")
        return
    last_used[uid] = now
    daily[day_key] += 1
    lang = ctx.user_data.get("lang", "ru")
    waiting += 1
    await msg.reply_text(f"Генерирую (в очереди: {waiting}), на CPU это может занять до минуты...")
    try:
        await _generate(ctx, msg, text, ref, lang)
    finally:
        waiting -= 1


async def _generate(ctx, msg, text, ref, lang):
    async with tts_lock:
        await ctx.bot.send_chat_action(msg.chat_id, ChatAction.RECORD_VOICE)
        with tempfile.TemporaryDirectory() as tmp:
            wav, ogg = Path(tmp) / "out.wav", Path(tmp) / "out.ogg"
            try:
                await asyncio.to_thread(synthesize, text, ref, lang, wav)
                await asyncio.to_thread(ffmpeg, "-i", str(wav), "-c:a", "libopus",
                                        "-b:a", "48k", str(ogg))
            except Exception:
                log.exception("synthesis failed")
                await msg.reply_text("Ошибка генерации, попробуй ещё раз.")
                return
            with ogg.open("rb") as fh:
                await msg.reply_voice(fh)


def main():
    global tts
    import torch
    from TTS.api import TTS
    log.info("Loading XTTS-v2...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
        "cuda" if torch.cuda.is_available() else "cpu")
    log.info("Model loaded")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("agree", agree))
    app.add_handler(CommandHandler("lang", set_lang))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, on_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
