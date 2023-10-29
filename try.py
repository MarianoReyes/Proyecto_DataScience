import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

# Define la arquitectura del modelo


class Deberta(nn.Module):
    def __init__(self, deberta_model):
        super(Deberta, self).__init__()
        self.deberta = deberta_model
        self.model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        x = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        x = x.logits  # Obtén las logits de la salida de Deberta
        x = self.model(x)
        return x


# Ruta al modelo preentrenado guardado localmente
modelo_preentrenado_path = "deberta_nlp.pth"

# Cargar el tokenizer del modelo preentrenado
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Crear una instancia del modelo Deberta
deberta_model = Deberta(
    PreTrainedModel.from_pretrained("microsoft/deberta-base"))

# Cargar los pesos del modelo desde el archivo .pth
deberta_model.load_state_dict(torch.load(
    modelo_preentrenado_path, map_location=torch.device('cpu')))

# Asegurarse de que el modelo esté en modo de evaluación
deberta_model.eval()

new_text = "Tu texto de ejemplo aquí..."

# Tokenizar y ajustar la secuencia de entrada
nuevo_resumen_encoded = tokenizer(
    new_text, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Realizar la predicción con el modelo
with torch.no_grad():
    input_ids = nuevo_resumen_encoded['input_ids']
    attention_mask = nuevo_resumen_encoded['attention_mask']

    outputs = deberta_model(input_ids, attention_mask)
    content_prediction, wording_prediction = outputs[0][0], outputs[0][1]

# Imprimir las predicciones
print(f'Predicción para "content": {content_prediction.item()}')
print(f'Predicción para "wording": {wording_prediction.item()}')
