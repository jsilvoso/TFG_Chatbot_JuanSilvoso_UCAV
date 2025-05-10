# Programa principal

from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections
import spacy
import os
import openai
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
from collections import deque
import csv
from flask import send_file
from datetime import datetime

load_dotenv()

# Inicializa Flask
app = Flask(__name__)

#Filtro para que la hora sea legible
@app.template_filter('timestamp_to_str')
def timestamp_to_str(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

# Configuración OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: No se encontró la clave API de OpenAI.")
    exit(1)
client = openai.OpenAI(api_key=api_key)

# Inicializa spaCy
try:
    import spacy.util
    if not spacy.util.is_package("es_core_news_sm"):
        print("Instalando modelo spaCy...")
        os.system("python -m spacy download es_core_news_sm")
except ImportError:
    os.system("pip install spacy==3.5.0")
    os.system("python -m spacy download es_core_news_sm")

nlp = spacy.load("es_core_news_sm")

# Inicializa recursos
process = psutil.Process(os.getpid())
metricas = deque(maxlen=100)

# Chatbot NLTK
pares = [
    (r"hola|buenas", ["\u00a1Hola! ¿En qué puedo ayudarte?", "\u00a1Hola!"]),
    (r"cómo estás", ["Estoy bien, gracias. ¿Y tú?", "Muy bien, ¿y tú?"]),
    (r"adiós|chao", ["\u00a1Hasta luego!", "Adiós, que tengas un buen día."])
]
chat_nltk = Chat(pares, reflections)

def get_best_match(user_input):
    responses = {
        "hola": "\u00a1Hola! ¿En qué puedo ayudarte?",
        "cómo estás": "Estoy bien, gracias por preguntar.",
        "adiós": "\u00a1Hasta luego!"
    }
    user_doc = nlp(user_input)
    best_match = max(responses.keys(), key=lambda x: nlp(x).similarity(user_doc))
    return responses[best_match] if user_doc.similarity(nlp(best_match)) > 0.5 else "No entendí tu pregunta."

# Transformers
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_transformer_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response if response else "No tengo una respuesta para eso."

# Guardar métricas
def guardar_metricas(nombre_modelo, inicio, fin):
    latency = round(fin - inicio, 4)
    cpu_usage = round(process.cpu_percent(interval=0.05), 2)
    memory_usage_mb = round(process.memory_info().rss / 1024 / 1024, 2)
    timestamp = time.time()

    metrica = {
        "modelo": nombre_modelo,
        "latencia": latency,
        "cpu": cpu_usage,
        "memoria": memory_usage_mb,
        "timestamp": timestamp
    }

    metricas.append(metrica)

    # Escribir en CSV
    archivo = "metricas.csv"
    archivo_nuevo = not os.path.exists(archivo)

    with open(archivo, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["modelo", "latencia", "cpu", "memoria", "timestamp"])
        if archivo_nuevo:
            writer.writeheader()
        writer.writerow(metrica)

# Rutas Flask
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat_nltk", methods=["POST"])
def chat_nltk_response():
    start = time.time()
    user_input = request.json.get("message", "")
    response = chat_nltk.respond(user_input) or "No entendí lo que dijiste."
    end = time.time()
    guardar_metricas("NLTK", start, end)
    return jsonify({"response": response})

@app.route("/chat_embeddings", methods=["POST"])
def chat_embeddings_response():
    start = time.time()
    user_input = request.json.get("message", "")
    response = get_best_match(user_input)
    end = time.time()
    guardar_metricas("Embeddings", start, end)
    return jsonify({"response": response})

@app.route("/chat_transformers", methods=["POST"])
def chat_transformers_response():
    start = time.time()
    user_input = request.json.get("message", "")
    response = generate_transformer_response(user_input)
    end = time.time()
    guardar_metricas("Transformers", start, end)
    return jsonify({"response": response})

def generate_openai_response(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un asistente últil."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"ERROR: {str(e)}"

@app.route("/chat_openai", methods=["POST"])
def chat_openai_response():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "ERROR: No se recibió un mensaje válido."})

    try:
        start = time.time()
        response = generate_openai_response(user_input)
        end = time.time()
        guardar_metricas("OpenAI", start, end)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"ERROR en servidor: {str(e)}"})

@app.route("/api/metricas")
def api_metricas():
    modelos = ["NLTK", "Embeddings", "OpenAI"]
    ultimas = {modelo: None for modelo in modelos}

    for metrica in reversed(metricas):
        if metrica["modelo"] in ultimas and ultimas[metrica["modelo"]] is None:
            ultimas[metrica["modelo"]] = metrica

    resultado = []
    for modelo in modelos:
        if ultimas[modelo]:
            resultado.append(ultimas[modelo])
        else:
            resultado.append({
                "modelo": modelo,
                "latencia": None,
                "cpu": None,
                "memoria": None,
                "timestamp": time.time()
            })

    return jsonify(resultado)

@app.route("/descargar_metricas")
def descargar_metricas():
    archivo = "metricas.csv"
    if not os.path.exists(archivo):
        # Si no existe, crea uno vacío con cabecera
        with open(archivo, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["modelo", "latencia", "cpu", "memoria", "timestamp"])
            writer.writeheader()
    return send_file(archivo, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)