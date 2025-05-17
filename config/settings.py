from dotenv import load_dotenv
from os import environ

load_dotenv()

# TELEGRAM NOTIFICATION
TELEGRAM_BOT_TOKEN = environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = environ.get('TELEGRAM_CHAT_ID')
