import cv2
import imageio.v2
from tkinter import filedialog
from tkinter import Tk

import numpy as np
import matplotlib.pyplot as plt

def import_image(N):
        """
        Importar imagen desde el administrador de archivos
        Returns:
            any: Imagen importada.
        """
        # Seleccionar archivo de imagen
        # abrir un cuadro de diálogo para seleccionar la imagen
        Tk().withdraw()  # evita que aparezca la ventana principal de Tkinter
        M = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        print(f"Archivo seleccionado: {M}")
        #cargar imagen desde las imagenes de grises
        img=cv2.imread(M,cv2.IMREAD_UNCHANGED)
        img=img[:, :, 0]  # Convertir a escala de grises si es una imagen RGB
        img = img / np.max(img) #normalizar la imagen para que los valores estén entre 0 y 1
        return img
def plot_image(img, title='Imagen', cmap_type='gray'):
    """
    Función para mostrar una imagen con un título y un mapa de colores específico.
    
    Args:
        img (array-like): La imagen a mostrar.
        title (str): El título de la imagen.
        cmap_type (str): El tipo de mapa de colores a usar (por ejemplo, 'gray', 'hot', 'viridis').
    """
    mag= np.abs(img)
    mag /= np.max(mag)



    plt.imshow(np.log(1+mag), cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()