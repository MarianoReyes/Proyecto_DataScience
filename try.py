import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/deberta")

# Crear una instancia del modelo
modelo_local = AutoModelForSequenceClassification.from_pretrained(
    "models/deberta")


# Cargar el archivo CSV
df = pd.read_csv("data/summaries_train.csv")

# Crear listas para almacenar los resultados
textos = []
contents = []
wordings = []
probabilidades_clase_0 = []
probabilidades_clase_1 = []

# Iterar sobre las primeras 10 filas del DataFrame
for index, row in df.head(100).iterrows():
    # Obtener el texto de la fila
    texto_de_prueba = row["text"]

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

    # Guardar los resultados
    textos.append(row["text"])
    contents.append(row["content"])
    wordings.append(row["wording"])
    probabilidades_clase_0.append(probabilidades_clase[0][0].item())
    probabilidades_clase_1.append(probabilidades_clase[0][1].item())

# Crear un nuevo DataFrame con los resultados
resultados_df = pd.DataFrame({
    "text": textos,
    "content": contents,
    "wording": wordings,
    "probabilidad_clase_0": probabilidades_clase_0,
    "probabilidad_clase_1": probabilidades_clase_1
})

# Guardar los resultados en un nuevo archivo CSV
resultados_df.to_csv("data/resultados_deberta.csv", index=False)
