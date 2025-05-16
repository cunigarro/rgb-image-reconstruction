from telegram import Bot

bot_token = '7248407303:AAEwITYB3KgY4Eff11Jhgyq5c8tC3bHVDkk'
chat_id = '6411041440'

bot = Bot(token=bot_token)
bot.send_message(chat_id=chat_id, text="✅ Entrenamiento HSCNN-D finalizado.")
