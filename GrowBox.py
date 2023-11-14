import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from matplotlib.dates import DateFormatter, HourLocator  # Importa HourLocator

# Cargar los datos desde el archivo CSV
ruta_archivo_csv = r'D:\IoT-Project\NDVI-Monitoring\PythonFotos01\growBox.csv'
df = pd.read_csv(ruta_archivo_csv)
# Convertir la columna 'Date' en datetime con el formato correcto
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y, %I:%M:%S %p')

# Filtrar las columnas para la primera gráfica (t1, h1)
columns_to_plot_temp_humidity = ['t1', 'h1']
# Filtrar las columnas para la segunda gráfica (co2)
columns_to_plot_co2 = ['co2']

# Manejar los valores NaN reemplazándolos por un valor específico (puede ser la media, mediana, etc.)
df[columns_to_plot_temp_humidity] = df[columns_to_plot_temp_humidity].fillna(df[columns_to_plot_temp_humidity].median())
df[columns_to_plot_co2] = df[columns_to_plot_co2].fillna(df[columns_to_plot_co2].median())

# Detección de anomalías utilizando Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
outliers = clf.fit_predict(df[columns_to_plot_temp_humidity + columns_to_plot_co2])

# Filtrar los datos para eliminar los outliers detectados
df_cleaned = df[outliers == 1]

# Etiquetas personalizadas para el eje Y de las gráficas
labels_temp_humidity = {'t1': 'Tемпература', 'h1': 'Влажность'}
labels_co2 = {'co2': 'Cо2'}

# Graficar las columnas de temperatura y humedad en una gráfica
plt.figure(figsize=(10, 6))
for column in columns_to_plot_temp_humidity:
    plt.plot(df_cleaned['Date'], df_cleaned[column], label=labels_temp_humidity[column])

plt.xlabel('Дата')  # Etiqueta para el eje X
plt.ylabel('Tемпература °C / Влажность %')  # Etiqueta para el eje Y
plt.grid(True)
# Ajustar el formato de fecha y el intervalo de las marcas de tiempo en el eje X
date_format = DateFormatter('(%Y-%m-%d) %H:%M')  # Formato de fecha: AAAA-MM-DD, HH:MM (24 horas)
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(HourLocator(interval=6))  # Mostrar una marca cada hora
plt.xticks(rotation=45)
plt.legend()
plt.title('Данные о Tемпературе и Влажность с 24.08.2023 До 29.08.2023')
plt.tight_layout()
plt.show()

# Graficar la columna de CO2 en otra gráfica
plt.figure(figsize=(10, 6))
for column in columns_to_plot_co2:
    plt.plot(df_cleaned['Date'], df_cleaned[column], label=labels_co2[column])

plt.xlabel('Дата')  # Etiqueta para el eje X
plt.ylabel('PPM')  # Etiqueta para el eje Y
plt.grid(True)
# Ajustar el formato de fecha y el intervalo de las marcas de tiempo en el eje X
date_format = DateFormatter('(%Y-%m-%d) %H:%M')  # Formato de fecha: AAAA-MM-DD, HH:MM (24 horas)
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(HourLocator(interval=6))  # Mostrar una marca cada hora
plt.xticks(rotation=45)
plt.legend()
plt.title('Данные о Cо2 с 24.08.2023 До 29.08.2023')
plt.tight_layout()
plt.show()

