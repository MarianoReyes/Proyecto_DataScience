import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/debertav3base")

# Crear una instancia del modelo
modelo_local = AutoModelForSequenceClassification.from_pretrained(
    "models/debertav3base")

# Texto de entrada para hacer una predicción (reemplaza con tu propio texto)
texto_de_prueba = "The Third Wave was an experiment to see how people reacted to a new one leader government."

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

# Ahora puedes acceder a las probabilidades para cada clase
# Probabilidad de la clase 0
probabilidad_clase_0 = probabilidades_clase[0][0].item()
# Probabilidad de la clase 1
probabilidad_clase_1 = probabilidades_clase[0][1].item()

print("Probabilidad de la clase 0:", probabilidad_clase_0)
print("Probabilidad de la clase 1:", probabilidad_clase_1)
