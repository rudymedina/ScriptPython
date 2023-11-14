import rasterio
import numpy as np
import matplotlib.pyplot as plt

def calcular_ndvi(imagen_red, imagen_nir):
    # Convertir las imágenes a arrays de numpy
    red = np.array(imagen_red, dtype=np.float32)
    nir = np.array(imagen_nir, dtype=np.float32)

    # Calcular el NDVI
    numerador = nir - red
    denominador = nir + red
    ndvi = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador != 0)

    return ndvi

def calcular_ndvi_b(imagen_red, imagen_blue):
    # Convertir las imágenes a arrays de numpy
    red = np.array(imagen_red, dtype=np.float32)
    blue = np.array(imagen_blue, dtype=np.float32)

    # Calcular el NDVI_B
    numerador = red - blue
    denominador = red + blue
    ndvi_b = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador != 0)

    return ndvi_b

def calcular_ndmi(imagen_red, imagen_nir, imagen_green):
    # Convertir las imágenes a arrays de numpy
    red = np.array(imagen_red, dtype=np.float32)
    nir = np.array(imagen_nir, dtype=np.float32)
    green = np.array(imagen_green, dtype=np.float32)

    # Calcular el NDMI
    numerador = nir - red
    denominador = nir + red
    ndmi = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador != 0)

    return ndmi

def calcular_ndre(imagen_red, imagen_reg):
    # Convertir las imágenes a arrays de numpy
    red = np.array(imagen_red, dtype=np.float32)
    reg = np.array(imagen_reg, dtype=np.float32)

    # Calcular el NDRE (corregido de 'nir' a 'reg')
    numerador = reg - red
    denominador = reg + red
    ndre = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador != 0)

    return ndre

def main():
    # Ruta y nombre de la imagen TIF de la banda RED (cambia la ruta y el nombre según tus archivos)
    ruta_imagen_red = r'D:\IoT-Project\NDVI-Monitoring\PythonFotos01\04_RED.TIF'

    # Ruta y nombre de la imagen TIF de la banda NIR (cambia la ruta y el nombre según tus archivos)
    ruta_imagen_nir = r'D:\IoT-Project\NDVI-Monitoring\PythonFotos01\04_NIR.TIF'

    # Ruta y nombre de la imagen TIF de la banda GRE (cambia la ruta y el nombre según tus archivos)
    ruta_imagen_gre = r'D:\IoT-Project\NDVI-Monitoring\PythonFotos01\04_GRE.TIF'

    # Ruta y nombre de la imagen TIF de la banda REG (cambia la ruta y el nombre según tus archivos)
    ruta_imagen_reg = r'D:\IoT-Project\NDVI-Monitoring\PythonFotos01\04_REG.TIF'

    # Cargamos las imágenes de las bandas RED, NIR, GRE y REG
    with rasterio.open(ruta_imagen_red) as src_red:
        imagen_red = src_red.read(1)

    with rasterio.open(ruta_imagen_nir) as src_nir:
        imagen_nir = src_nir.read(1)

    with rasterio.open(ruta_imagen_gre) as src_gre:
        imagen_gre = src_gre.read(1)

    with rasterio.open(ruta_imagen_reg) as src_reg:
        imagen_reg = src_reg.read(1)

    # Calcular los índices de vegetación (NDVI, NDVI_B, NDMI y NDRE)
    ndvi = calcular_ndvi(imagen_red, imagen_nir)
    ndvi_b = calcular_ndvi_b(imagen_red, imagen_gre)
    ndmi = calcular_ndmi(imagen_red, imagen_nir, imagen_gre)
    ndre = calcular_ndre(imagen_red, imagen_reg)

    # Crear una única figura y agregar todas las gráficas en subplots
    plt.figure(figsize=(16, 12))

    plt.subplot(221)
    plt.imshow(ndvi, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI')

    plt.subplot(222)
    plt.imshow(ndvi_b, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI_B')
    plt.title('NDVI_B')

    plt.subplot(223)
    plt.imshow(ndmi, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar(label='NDMI')
    plt.title('NDMI')

    plt.subplot(224)
    plt.imshow(ndre, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar(label='NDRE')
    plt.title('NDRE')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()