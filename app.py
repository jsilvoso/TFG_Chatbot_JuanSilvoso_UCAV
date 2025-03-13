#Programa principal
#Primera Versión: "Se genera una respuesta si la pregunta coincide con alguna de las establecidas"

from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections

import spacy
import os
#import torch

#Fuerza la instalación de versiones compatibles de numpy y spacy para solucionar el error que aparecía.
os.system("pip install --upgrade --force-reinstall numpy==1.23.5 spacy thinc")

# Descargar el modelo si no existe
os.system("python -m spacy download es_core_news_sm")

# Cargar el modelo
nlp = spacy.load("es_core_news_sm")
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
"""
app = Flask(__name__)

# ---- Chatbot NLTK ---- #
pares = [
    (r"hola|buenas", ["¡Hola! ¿En qué puedo ayudarte?", "¡Hola!"]),
    (r"cómo estás", ["Estoy bien, gracias. ¿Y tú?", "Muy bien, ¿y tú?"]),
    (r"adiós|chao", ["¡Hasta luego!", "Adiós, que tengas un buen día."])
]
chat_nltk = Chat(pares, reflections)

# ---- Chatbot con Embeddings (spaCy) ---- #
nlp = spacy.load("es_core_news_sm")
def get_best_match(user_input):
    responses = {
        "hola": "¡Hola! ¿En qué puedo ayudarte?",
        "cómo estás": "Estoy bien, gracias por preguntar.",
        "adiós": "¡Hasta luego!"
    }
    user_doc = nlp(user_input)
    best_match = max(responses.keys(), key=lambda x: nlp(x).similarity(user_doc))
    return responses[best_match] if user_doc.similarity(nlp(best_match)) > 0.5 else "No entendí tu pregunta."
"""
# ---- Chatbot con Transformers ---- #
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_transformer_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response if response else "No tengo una respuesta para eso."
"""
# ---- Rutas Flask ---- #
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat_nltk", methods=["POST"])
def chat_nltk_response():
    user_input = request.json.get("message", "")
    response = chat_nltk.respond(user_input) or "No entendí lo que dijiste."
    return jsonify({"response": response})

@app.route("/chat_embeddings", methods=["POST"])
def chat_embeddings_response():
    user_input = request.json.get("message", "")
    response = get_best_match(user_input)
    return jsonify({"response": response})
"""
@app.route("/chat_transformers", methods=["POST"])
def chat_transformers_response():
    user_input = request.json.get("message", "")
    response = generate_transformer_response(user_input)
    return jsonify({"response": response})

"""
if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 10000))  # Puerto de Render
    app.run(host="0.0.0.0", port=5000)

#Prueba hecha en entorno virual correctamente