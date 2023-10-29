import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/deberta")

# Crear una instancia del modelo
modelo_local = AutoModelForSequenceClassification.from_pretrained(
    "models/deberta")

# Texto de entrada para hacer una predicción (reemplaza con tu propio texto)
texto_de_prueba = "In the story ""The Jungle"" written by Upton Sinclair, members of a factory would sell and use spoiled meat for money. The factory would do several things to cover up the spoiled meat, they rubbed soda on it to take away the smell. They also used pickles to destroy the odor. ""the packers had a second and much stronger pickle which destroyed the oder"". Basically, they did whatever they could to sell this meat. "

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
