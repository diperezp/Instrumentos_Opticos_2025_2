import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt


def import_image():
    """
    Esta funcion abre una pantalla emergente del administrador de archivos para la ruta de una imagen a importar.
    """
    # Ocultar la ventana principal de Tkinter
    Tk().withdraw()

    # Abrir el cuadro de diálogo para seleccionar la imagen
    ruta_imagen = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )

    # Cargar la imagen como un arreglo numpy
    if ruta_imagen:
        #cargamos el primer 
        imagen = Image.open(ruta_imagen)
        return np.array(imagen)
    else:
        return None

def show_image(img_array):
    """
    Esta funcion muestra una imagen dada como un arreglo numpy.
    """
    plt.imshow(img_array, cmap='gray', origin='upper')
    plt.axis('off')  # Ocultar los ejes
    plt.show()

def export_image(img_array):
    """
    Esta funcion abre una pantalla emergente del administrador de archivos para guardar una imagen.
    """
    # Ocultar la ventana principal de Tkinter
    Tk().withdraw()

    # Abrir el cuadro de diálogo para guardar la imagen
    ruta_guardado = filedialog.asksaveasfilename(
        title="Guardar imagen",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tiff")]
    )

    # Guardar la imagen si se proporcionó una ruta
    if ruta_guardado:
        imagen = Image.fromarray(img_array)
        imagen.save(ruta_guardado)

