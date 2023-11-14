import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from matplotlib.dates import DateFormatter, HourLocator

# Cargar los datos desde el archivo CSV
ruta_archivo_csv = r'D:\IoT-Project\NDVI-Monitoring\PythonFotos01\reportStorage.csv'
df = pd.read_csv(ruta_archivo_csv)
# Convertir la columna 'Date' en datetime con el formato correcto
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y, %I:%M:%S %p')



# Filtrar las columnas para la primera gráfica (t1, t2, t3)
#columns_to_plot_temp = ['t1', 't2', 't3']
columns_to_plot_temp = ['t1', 't2', 't3', 't4']
#columns_to_plot_temp = ['t1', 't2']

# Manejar los valores NaN reemplazándolos por un valor específico (puede ser la media, mediana, etc.)
df[columns_to_plot_temp] = df[columns_to_plot_temp].fillna(df[columns_to_plot_temp].median())

# Detección de anomalías utilizando Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
outliers = clf.fit_predict(df[columns_to_plot_temp])

# Filtrar los datos para eliminar los outliers detectados
df_cleaned = df[outliers == 1]

# Etiquetas personalizadas para el eje Y de la primera gráfica
#labels_temp = {'t1': 'Температура 1', 't2': 'Температура 2', 't3': 'Температура 3'}
labels_temp = {'t1': 'ср Температура', 't2': 'Температура 1', 't3': 'Температура 2', 't4': 'Температура 3'}
#labels_temp = {'t1': 'СР температуре Склад 1', 't2': 'СР температуре Склад 2'}

# Graficar las columnas de temperatura en una sola gráfica
plt.figure(figsize=(10, 6))
for column in columns_to_plot_temp:
    plt.plot(df_cleaned['Date'], df_cleaned[column], label=labels_temp[column])

plt.xlabel('Дата')  # Etiqueta para el eje X
plt.ylabel('Температура')  # Etiqueta para el eje Y
plt.grid(True)
# Ajustar el formato de fecha y el intervalo de las marcas de tiempo en el eje X
date_format = DateFormatter('(%Y-%m-%d) %H:%M')  # Formato de fecha: AAAA-MM-DD, HH:MM (24 horas)
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(HourLocator(interval=1))  # Mostrar una marca cada hora
plt.xticks(rotation=45)
plt.legend()
#plt.title('Данные мониторинга последние 24 часа')
plt.title('Данные о температуре с 29.08.2023 До 30.08.2023')
plt.tight_layout()
plt.show()
