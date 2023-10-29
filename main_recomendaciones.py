from flask import Flask, request, render_template
import torch
import torch.nn as nn
from transformers import AutoTokenizer

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process_text_deberta', methods=['POST'])
def process_text_deberta():
    if request.method == 'POST':
        new_text = request.form['input_text']
        # Tokenizar y ajustar la secuencia de entrada

        # aqui realizar la prediccion

        # content=predicted_content, wording=predicted_wording, show_content_tips=show_content_tips, show_wording_tips=show_wording_tips, show_success_content=show_success_content, show_success_wording=show_success_wording
        return render_template('results_deberta.html', input_text=new_text)


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
