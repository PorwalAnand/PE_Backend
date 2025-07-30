from pymongo import MongoClient
from app.config import MONGO_URI, MONGO_DB

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

chat_history = db["chat_history"]
model_usage = db["model_usage"]

