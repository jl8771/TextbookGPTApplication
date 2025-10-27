from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import ChatSession
import json

app = Flask(__name__)
CORS(app)
chat_session = ChatSession("qwen")

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/start', methods=['POST'])
def start_chat():
    query = request.form.get('query')
    res = chat_session.start_chat(query)
    return jsonify({"answer": res})

@app.route('/continue', methods=['POST'])
def continue_chat():
    query = request.form.get('query', '')
    res = chat_session.continue_chat(query)
    return jsonify({"answer": res})

@app.route('/clear', methods=['POST'])
def clear_chat():
    chat_session.clear_chat()
    return jsonify({"status": "ok"})

@app.route('/provider', methods=['POST'])
def get_provider():
    provider = chat_session.get_provider()
    return jsonify({"provider": provider})