import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from matplotlib.dates import DateFormatter, HourLocator

# Cargar los datos desde el archivo CSV
ruta_archivo_csv = r'C:\Users\user\Documents\GitScripts\growBox.csv'
df = pd.read_csv(ruta_archivo_csv)
# Convertir la columna 'Date' en datetime con el formato correcto
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y, %I:%M:%S %p')

# Filtrar las columnas para la primera gráfica (t1, t2, t3)
columns_to_plot_temp = ['t1', 't2', 't3']

# Manejar los valores NaN reemplazándolos por un valor específico (puede ser la media, mediana, etc.)
df[columns_to_plot_temp] = df[columns_to_plot_temp].fillna(df[columns_to_plot_temp].median())

# Detección de anomalías utilizando Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
outliers = clf.fit_predict(df[columns_to_plot_temp])

# Filtrar los datos para eliminar los outliers detectados
df_cleaned = df[outliers == 1]

# Etiquetas personalizadas para el eje Y de la primera gráfica
labels_temp = {'t1': 'Co2 камера 1', 't2': 'Co2 камера 2', 't3': 'Co2 камера 3'}

# Graficar las columnas de temperatura en una sola gráfica
plt.figure(figsize=(10, 6))
for column in columns_to_plot_temp:
    plt.plot(df_cleaned['Date'], df_cleaned[column], label=labels_temp[column])

plt.xlabel('Дата')  # Etiqueta para el eje X
plt.ylabel('Влажность')  # Etiqueta para el eje Y
plt.yticks(np.arange(300, 1300, step=100))  # Definir los valores en el eje Y
plt.grid(True, alpha=0.5)  # Cuadrículas transparentes
# Ajustar el formato de fecha y el intervalo de las marcas de tiempo en el eje X
date_format = DateFormatter('(%Y-%m-%d) %H:%M')  # Formato de fecha: AAAA-MM-DD, HH:MM (24 horas)
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(HourLocator(interval=6))  # Mostrar una marca cada hora
plt.xticks(rotation=45)
plt.legend()
plt.title('Данные о Влажность с 10 Ноябрь 2023 До 15 Ноябрь 2023')
plt.tight_layout()
plt.show()