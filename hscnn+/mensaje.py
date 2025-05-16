import asyncio
from telegram import Bot

async def notify():
    bot_token = '7248407303:AAEwITYB3KgY4Eff11Jhgyq5c8tC3bHVDkk'
    chat_id = '6411041440'
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text="✅ Entrenamiento HSCNN-D finalizado.")

# Al final de tu script:
asyncio.run(notify())
