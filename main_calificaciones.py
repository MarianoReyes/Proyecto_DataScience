from flask import Flask, request, render_template
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import percentileofscore
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from translate import Translator
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import enchant
from math import log
import re

# Cargar el modelo GPT-2 y el tokenizador
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Función para dividir un texto en fragmentos de tamaño máximo
def split_text(text, max_length):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def herdan_index(text):
    unique_words = set(text)
    word_count = len(text)
    diversity_index = len(unique_words) / log(word_count)
    return diversity_index

def yules_k(text):
    total_words = len(text)
    word_freq = nltk.FreqDist(text)
    m1 = sum([freq**2 for freq in word_freq.values()])
    m2 = sum([freq for freq in word_freq.values()]) ** 2

    k = 10000 * (m1 - m2) / (m2 - total_words)
    return k


# Cargar el tokenizer
with open('./models/lstm/tokenizer.pickle', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Cargar el modelo
model = load_model('./models/lstm/modelo_resumen.h5')

def preprocess_text(text):
    # Tokeniza el texto
    text_sequence = tokenizer.texts_to_sequences([text])
    padded_text_sequence = pad_sequences(text_sequence, maxlen=100)
    return padded_text_sequence


app = Flask(__name__)

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('data/summaries_train.csv')

# Encontrar el valor máximo y mínimo en la columna 'content'
max_content = df['content'].max()
min_content = df['content'].min()

# Encontrar el valor máximo y mínimo en la columna 'wording'
max_wording = df['wording'].max()
min_wording = df['wording'].min()

@app.route('/')
def home():
    return render_template('profesores/index.html')


@app.route('/process_text_deberta', methods=['POST'])
def process_text_deberta():
    if request.method == 'POST':
        texto_de_prueba = request.form['input_text']

        show_content_tips = False
        show_success_content = False
        show_wording_tips = False
        show_success_wording = False
        show_token_tips = False
        show_success_token = False

        # Tokenizar y ajustar la secuencia de entrada

        # Cargar el tokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/debertav3base")

        # Crear una instancia del modelo
        modelo_local = AutoModelForSequenceClassification.from_pretrained(
            "models/debertav3base")

        # Tokeniza el texto
        tokens = tokenizer(texto_de_prueba, padding=True,
                           truncation=True, max_length=512, return_tensors="pt")

        # Realiza la predicción
        with torch.no_grad():
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            prediction = modelo_local(input_ids, attention_mask)

        # Obtener las probabilidades de clase
        probabilidades_clase = torch.softmax(prediction.logits, dim=1)

        predicted_content = probabilidades_clase[0][0].item()
        predicted_wording = probabilidades_clase[0][1].item()

        # Definir el nuevo número que deseas comparar
        nuevo_numero_content = predicted_content
        nuevo_numero_wording = predicted_wording

        # Calcular el porcentaje para 'content'
        predicted_content = (nuevo_numero_content -
                             min_content) / (max_content - min_content)

        # Calcular el porcentaje para 'wording'
        predicted_wording = (nuevo_numero_wording -
                             min_wording) / (max_wording - min_wording)

        if predicted_content < 0.6:
            show_content_tips = True
        else:
            show_success_content = True

        if predicted_wording < 0.7:
            show_wording_tips = True
        else:
            show_success_wording = True

        # Calculate percentiles
        percentile_content = percentileofscore(df['content'], predicted_content)
        percentile_wording = percentileofscore(df['wording'], predicted_wording)

        # elementos especificos de profesor

        words = word_tokenize(texto_de_prueba.lower())  # Lowercasing for consistency
        # Calculate lexical diversity using the type-token ratio
        type_token_ratio = len(set(words)) / len(words)
        type_token_ratio = round(type_token_ratio, 2)

        if type_token_ratio < 0.6:
            show_token_tips = True
        else:
            show_success_token = True

        score_sugested = (((predicted_content + predicted_wording + type_token_ratio) / 3) + 0.2) * 100
        score_sugested = round(score_sugested, 2)
        if score_sugested > 100:
            score_sugested = 100

        # Tokenize sentences and words
        sentences = sent_tokenize(texto_de_prueba)
        words = word_tokenize(texto_de_prueba)

        # Calculate average sentence length and word length
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Total number of words and sentences
        total_words = len(words)
        total_sentences = len(sentences)

        # Create an English dictionary object
        english_dict = enchant.Dict("en_US")

        # Obtener palabras y filtrar símbolos usando expresiones regulares
        words = re.findall(r'\b\w+\b', texto_de_prueba)

        # errores ortograficos
        spelling_errors = [word for word in words if not english_dict.check(word)]

        return render_template('profesores/results_deberta.html', 
                               input_text=texto_de_prueba, 
                               content=predicted_content, 
                               wording=predicted_wording, 
                               type_token_ratio=type_token_ratio, 
                               show_token_tips=show_token_tips, 
                               show_content_tips=show_content_tips, 
                               show_wording_tips=show_wording_tips, 
                               show_success_token=show_success_token, 
                               show_success_content=show_success_content, 
                               show_success_wording=show_success_wording, 
                               percentile_content=percentile_content, 
                               percentile_wording=percentile_wording, 
                               score_sugested=score_sugested, 
                               spelling_errors=spelling_errors,
                               avg_sentence_length=avg_sentence_length,
                               avg_word_length=avg_word_length,
                               total_words=total_words,
                               total_sentences=total_sentences)

@app.route('/process_text_lstm', methods=['POST'])
def process_text_lstm():
    if request.method == 'POST':
        new_text = request.form['input_text']

        show_content_tips = False
        show_success_content = False
        show_wording_tips = False
        show_success_wording = False
        show_token_tips = False
        show_success_token = False

        # Preprocesa el texto de entrada
        padded_text_sequence = preprocess_text(new_text)

        # Realiza la predicción
        prediction = model.predict(padded_text_sequence)

        # La predicción contendrá los puntajes de contenido y redacción
        predicted_content = prediction[0][0]
        predicted_wording = prediction[0][1]

        # Determina si se deben mostrar recomendaciones
        show_content_tips = predicted_content < 0.6
        show_wording_tips = predicted_wording < 0.7

        if predicted_content < 0.6:
            show_content_tips = True
        else:
            show_success_content = True

        if predicted_wording < 0.7:
            show_wording_tips = True
        else:
            show_success_wording = True

        # Calculate percentiles
        percentile_content = percentileofscore(df['content'], predicted_content)
        percentile_wording = percentileofscore(df['wording'], predicted_wording)


        # elementos especificos de profesor

        words = word_tokenize(new_text.lower())  # Lowercasing for consistency
        # Calculate lexical diversity using the type-token ratio
        type_token_ratio = len(set(words)) / len(words)
        type_token_ratio = round(type_token_ratio, 2)

        if type_token_ratio < 0.6:
            show_token_tips = True
        else:
            show_success_token = True

        score_sugested = (((predicted_content + predicted_wording + type_token_ratio) / 3) + 0.2) * 100
        score_sugested = round(score_sugested, 2)
        if score_sugested > 100:
            score_sugested = 100

        # Tokenize sentences and words
        sentences = sent_tokenize(new_text)
        words = word_tokenize(new_text)

        # Calculate average sentence length and word length
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Total number of words and sentences
        total_words = len(words)
        total_sentences = len(sentences)

        # Create an English dictionary object
        english_dict = enchant.Dict("en_US")

        # Obtener palabras y filtrar símbolos usando expresiones regulares
        words = re.findall(r'\b\w+\b', new_text)

        # errores ortograficos
        spelling_errors = [word for word in words if not english_dict.check(word)]

        return render_template('profesores/results_lstm.html', 
                               input_text=new_text, 
                               content=predicted_content, 
                               wording=predicted_wording, 
                               type_token_ratio=type_token_ratio, 
                               show_token_tips=show_token_tips, 
                               show_content_tips=show_content_tips, 
                               show_wording_tips=show_wording_tips, 
                               show_success_token=show_success_token, 
                               show_success_content=show_success_content, 
                               show_success_wording=show_success_wording, 
                               percentile_content=percentile_content, 
                               percentile_wording=percentile_wording, 
                               score_sugested=score_sugested, 
                               spelling_errors=spelling_errors,
                               avg_sentence_length=avg_sentence_length,
                               avg_word_length=avg_word_length,
                               total_words=total_words,
                               total_sentences=total_sentences)

if __name__ == '__main__':
    app.run(debug=True)
