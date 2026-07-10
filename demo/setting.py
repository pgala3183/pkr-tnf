from flask_socketio import SocketIO
from flask import Flask
import os
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from app.game_settings import BB, SB, show_game, time_pause_round_end_one_winner, time_pause_round_end_split

# Create app flask
app = Flask(__name__)
socketio = SocketIO(app)

# Re-export game constants for legacy imports from setting.py

# poker-transformer FastAPI service (Step 9)
TRANSFORMER_API_URL = os.environ.get("TRANSFORMER_API_URL", "http://localhost:8000")


# A dictionary that stores all actively playing clients. "sid number" : "game instance"
games = {}

# Here login to your OpenAI account. You need key to API. To get access to key can use Azure key vault, another service,
# or just write key
"""
key_vault_url = "https://apike.vault.azure.net/"
secret_name = "OpenAI"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=key_vault_url, credential=credential)
openai_api_key = client.get_secret(secret_name).value
"""
# without azure key vault:
openai_api_key = 'xxx'

openai.api_key = openai_api_key




