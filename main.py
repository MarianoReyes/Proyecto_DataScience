from flask import Flask, request, render_template
import torch
import torch.nn as nn
from transformers import AutoTokenizer

app = Flask(__name__)

# Define la arquitectura del modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Deberta(nn.Module):
    def __init__(self, deberta):
        super(Deberta, self).__init__()
        self.deberta = deberta
        self.model = nn.Sequential(nn.Dropout(0.1),
                                   nn.Linear(2, 768),
                                   nn.ReLU(),
                                   nn.Linear(768, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask):
        x = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        x = x[0].type(torch.float32)
        x = self.model(x)
        return x


# Especifica la ruta o el nombre del modelo preentrenado que deseas utilizar
modelo_preentrenado = "/models/deberta"

# Carga el tokenizer del modelo preentrenado
tokenizer = AutoTokenizer.from_pretrained(modelo_preentrenado)

# Crea una instancia del modelo
# Puedes inicializarlo con cualquier modelo necesario
model = Deberta(deberta=None)

# Carga los pesos del modelo desde el archivo .pth
model.load_state_dict(torch.load('deberta_nlp.pth'))

# Asegúrate de que el modelo esté en modo de evaluación
model.eval()


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
