#Programa principal
#Primera Versión: "Se genera una respuesta si la pregunta coincide con alguna de las establecidas"

from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections
#import spacy #Comprobar fallos
import os
#import torch
import openai
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()  # Cargar variables desde el archivo .env

# Obtener clave API de OpenAI
api_key = os.getenv("OPENAI_API_KEY")

# Crear cliente OpenAI
#client = openai.OpenAI(api_key=api_key)

# Comprobación de la clave API
if not api_key:
    print("⚠️ ERROR: No se encontró la clave API de OpenAI en las variables de entorno.")
    exit(1)  # Sale del programa si la clave no está definida

openai.api_key = api_key
print(f"🔹 Clave API de OpenAI detectada: {openai.api_key[:10]}********")

#Importa torch si es necesario
try:
    import torch
except ImportError:
    print("Advertencia: torch no está instalado. Algunas funcionalidades pueden no estar disponibles.")

"""
try:
    import spacy
except ImportError:
    print("spaCy no está instalado. Instalándolo ahora...")
    os.system("pip install --no-cache-dir spacy==3.5.0")
    import spacy


# Verificar si el modelo está instalado antes de cargarlo
import spacy.util
if not spacy.util.is_package("es_core_news_sm"):
    print("El modelo de spaCy no está instalado. Instálalo manualmente con:")
    print("    python -m spacy download es_core_news_sm")
    os.system("python -m spacy download es_core_news_sm") #Para instalarlo
    exit(1)  # Salir del programa si el modelo no está instalado

# Carga el modelo
nlp = spacy.load("es_core_news_sm")

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
#nlp = spacy.load("es_core_news_sm")
def get_best_match(user_input):
    responses = {
        "hola": "¡Hola! ¿En qué puedo ayudarte?",
        "cómo estás": "Estoy bien, gracias por preguntar.",
        "adiós": "¡Hasta luego!"
    }
    user_doc = nlp(user_input)
    best_match = max(responses.keys(), key=lambda x: nlp(x).similarity(user_doc))
    return responses[best_match] if user_doc.similarity(nlp(best_match)) > 0.5 else "No entendí tu pregunta."

# ---- Chatbot con Transformers ---- #
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_transformer_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response if response else "No tengo una respuesta para eso."

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

@app.route("/chat_transformers", methods=["POST"])
def chat_transformers_response():
    user_input = request.json.get("message", "")
    response = generate_transformer_response(user_input)
    return jsonify({"response": response})

def generate_openai_response(user_input):
    try:
        response = openai.completions.create(
            model="gpt-4",  # Usa el modelo adecuado para tu cuenta
            prompt=f"Eres un asistente útil. Responde a la siguiente pregunta: {user_input}",
            max_tokens=100,
            temperature=0.7  # Puedes ajustar la aleatoriedad de la respuesta
        )
        return response['choices'][0]['text'].strip()
        #return response['choices'][0]['message']['content'].strip()
    except openai.OpenAIError as e:
        return f"❌ ERROR: {str(e)}"

# Nueva ruta Flask para el chatbot con OpenAI
@app.route("/chat_openai", methods=["POST"])
def chat_openai_response():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "⚠️ ERROR: No se recibió un mensaje válido."})

    try:
        response = generate_openai_response(user_input)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"❌ ERROR en servidor: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto de Render o 5000 por defecto
    app.run(host="0.0.0.0", port=port)
