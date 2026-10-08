import os
import logging
import tempfile
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# === НАСТРОЙКИ ===
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ОТ_BOTFATHER"  # Замените на свой токен
HF_TOKEN = os.getenv("HF_TOKEN")  # Токен Hugging Face (если модель требует)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальное хранилище модели и образцов голоса
model = None
user_voice_samples = {}  # {user_id: {"path": "...", "ref_text": "..."}}


def load_model():
    """Загружает модель Qwen3-TTS при старте."""
    global model
    logger.info("Загрузка модели Qwen3-TTS...")
    
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    logger.info("Модель загружена!")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start."""
    await update.message.reply_text(
        "👋 Привет! Я бот для клонирования голоса через Qwen3-TTS.\n\n"
        "📋 **Как использовать:**\n"
        "1. Отправь мне голосовое сообщение (образец голоса, 3+ секунды)\n"
        "2. Затем отправь текст, который нужно озвучить\n"
        "3. Я верну аудио с клонированным голосом\n\n"
        "⚠️ **Важно:** Образец должен содержать чистую речь без шума.\n"
        "Для сброса отправь /reset",
        parse_mode="Markdown"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сброс сохранённого образца голоса."""
    user_id = update.effective_user.id
    if user_id in user_voice_samples:
        del user_voice_samples[user_id]
    await update.message.reply_text("✅ Образец голоса сброшен. Отправь новый.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голосового сообщения — сохранение образца."""
    user_id = update.effective_user.id
    voice = update.message.voice
    
    # Скачиваем голосовое сообщение
    file = await context.bot.get_file(voice.file_id)
    temp_dir = Path(tempfile.gettempdir())
    sample_path = temp_dir / f"sample_{user_id}.wav"
    
    await file.download_to_drive(sample_path)
    
    # Сохраняем путь (текст образца можно оставить пустым для x_vector_only режима)
    user_voice_samples[user_id] = {
        "path": str(sample_path),
        "ref_text": ""  # Пустой текст = x_vector_only режим (только эмбеддинг голоса)
    }
    
    await update.message.reply_text(
        "✅ Образец голоса сохранён!\n"
        "Теперь отправь текст, который нужно озвучить этим голосом."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текста — генерация аудио."""
    user_id = update.effective_user.id
    text = update.message.text
    
    if user_id not in user_voice_samples:
        await update.message.reply_text(
            "❌ Сначала отправь голосовое сообщение как образец голоса!"
        )
        return
    
    if len(text) > 500:
        await update.message.reply_text("❌ Текст слишком длинный (макс. 500 символов).")
        return
    
    sample = user_voice_samples[user_id]
    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / f"output_{user_id}.wav"
    
    await update.message.reply_text("🎙️ Генерирую аудио... Это может занять несколько секунд.")
    
    try:
        # Генерация с клонированием голоса
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Russian",  # Можно изменить на "English", "Chinese" и т.д.
            ref_audio=sample["path"],
            ref_text=sample["ref_text"],
            x_vector_only_mode=True,  # True = используем только эмбеддинг голоса (не нужен текст образца)
        )
        
        # Сохраняем результат
        sf.write(str(output_path), wavs[0], sr)
        
        # Отправляем голосовое сообщение
        with open(output_path, 'rb') as audio:
            await update.message.reply_voice(voice=audio)
        
        # Очистка
        output_path.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")


def main():
    """Запуск бота."""
    # Загружаем модель
    load_model()
    
    # Создаём приложение
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Регистрируем хендлеры
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    logger.info("Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
