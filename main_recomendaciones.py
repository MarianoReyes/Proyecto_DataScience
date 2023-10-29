from flask import Flask, request, render_template
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process_text_deberta', methods=['POST'])
def process_text_deberta():
    if request.method == 'POST':
        texto_de_prueba = request.form['input_text']
        print(type(texto_de_prueba))
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

        # , show_content_tips=show_content_tips, show_wording_tips=show_wording_tips, show_success_content=show_success_content, show_success_wording=show_success_wording
        return render_template('results_deberta.html', input_text=texto_de_prueba, content=predicted_content, wording=predicted_wording)


@app.route('/process_text_lstm', methods=['POST'])
def process_text_lstm():
    if request.method == 'POST':
        new_text = request.form['input_text']
        # Tokenizar y ajustar la secuencia de entrada

        # aqui realizar la prediccion

        # content=predicted_content, wording=predicted_wording, show_content_tips=show_content_tips, show_wording_tips=show_wording_tips, show_success_content=show_success_content, show_success_wording=show_success_wording
        return render_template('results_lstm.html', input_text=new_text)


if __name__ == '__main__':
    app.run(debug=True)
