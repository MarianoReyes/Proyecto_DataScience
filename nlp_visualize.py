from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#model = tf.keras.models.load_model('nlp_gpt2.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        new_text = request.form['input_text']
        # Tokenizar y ajustar la secuencia de entrada
        new_encodings = tokenizer(
            new_text, truncation=True, padding='max_length', max_length=25, return_tensors='tf')
        # Realizar predicciones para contenido y estilo
        predicted_content, predicted_wording = model.predict(
            new_encodings['input_ids'])

        # Definir umbrales
        content_threshold = 0.6
        wording_threshold = 0.7

        # Determinar si se deben mostrar consejos
        show_content_tips = predicted_content < content_threshold
        show_success_content = predicted_content > content_threshold
        show_wording_tips = predicted_wording < wording_threshold
        show_success_wording = predicted_wording > wording_threshold

        return render_template('results.html', input_text=new_text, content=predicted_content, wording=predicted_wording, show_content_tips=show_content_tips, show_wording_tips=show_wording_tips, show_success_content=show_success_content, show_success_wording=show_success_wording)


if __name__ == '__main__':
    app.run(debug=True)
