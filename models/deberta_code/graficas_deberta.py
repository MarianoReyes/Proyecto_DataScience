import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('./data/resultados_deberta.csv')

# Seleccionar las columnas a normalizar
columns_to_normalize = ["actual_content", "actual_wording",
                        "predicted_content", "predicted_wording"]

# Inicializar el MinMaxScaler
scaler = MinMaxScaler()

# Normalizar las columnas seleccionadas
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Guardar el DataFrame normalizado en un nuevo archivo CSV o sobrescribir el original
df.to_csv('./data/resultados_deberta_normalizads.csv', index=False)


# Cargar el archivo CSV normalizado en un DataFrame
df = pd.read_csv('./data/resultados_deberta_normalizads.csv')

# Crear un scatter plot para comparar las predicciones con los valores reales de contenido
plt.figure(figsize=(8, 6))
plt.scatter(df['actual_content'], df['predicted_content'])
plt.title("Comparación de Predicciones vs. Valores Reales de Contenido")
plt.xlabel("Valores Reales de Contenido")
plt.ylabel("Predicciones de Contenido")
plt.grid(True)
plt.show()

# Crear un scatter plot para comparar las predicciones con los valores reales de redacción
plt.figure(figsize=(8, 6))
plt.scatter(df['actual_wording'], df['predicted_wording'])
plt.title("Comparación de Predicciones vs. Valores Reales de Redacción")
plt.xlabel("Valores Reales de Redacción")
plt.ylabel("Predicciones de Redacción")
plt.grid(True)
plt.show()

# Crear histogramas de errores para contenido
content_errors = df['actual_content'] - df['predicted_content']
plt.figure(figsize=(8, 6))
plt.hist(content_errors, bins=30, color='blue', alpha=0.7)
plt.title("Histograma de Errores para Contenido")
plt.xlabel("Errores")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

# Crear histogramas de errores para redacción
wording_errors = df['actual_wording'] - df['predicted_wording']
plt.figure(figsize=(8, 6))
plt.hist(wording_errors, bins=30, color='red', alpha=0.7)
plt.title("Histograma de Errores para Redacción")
plt.xlabel("Errores")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
