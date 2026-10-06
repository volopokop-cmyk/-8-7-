# Voice Clone Telegram Bot

Telegram-бот клонирования голоса на [Coqui XTTS-v2](https://github.com/idiap/coqui-ai-TTS).
Присылаешь голосовое (5-30 сек) -> присылаешь текст -> получаешь озвучку этим голосом.

Модель XTTS-v2 распространяется по лицензии CPML (только некоммерческое использование).
Клонируйте только голоса, на которые есть согласие владельца.

## Запуск через GitHub Actions
1. Создай бота у [@BotFather](https://t.me/BotFather), скопируй токен.
2. Репозиторий -> Settings -> Secrets and variables -> Actions -> **Secrets** -> `TELEGRAM_BOT_TOKEN`.
3. (Необязательно) **Variables** -> `ALLOWED_USER_IDS` - если задать, бот станет приватным. Если не задавать - бот публичный, им может пользоваться любой.
4. Вкладка Actions -> "Run voice bot" -> Run workflow. Дальше бот перезапускается сам каждые 6 часов.

## Локально
```bash
pip install torch torchaudio && pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=... COQUI_TOS_AGREED=1
python bot.py
```

## Публичный режим
Без `ALLOWED_USER_IDS` бот открыт всем. Защита от перегрузки: очередь (макс. 4), пауза 30 сек между
запросами, 15 генераций в сутки на пользователя, текст до 500 символов, обязательное `/agree`.
Лимиты настраиваются константами в начале `bot.py`.
