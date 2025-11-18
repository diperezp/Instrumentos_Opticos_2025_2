import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt
from utilopctic import *
import cv2
from LightPipes import *

def rescale_field(F, new_size, wavelength):
    """
    Reescala un campo de LightPipes a un nuevo tamaño físico.
    
    Parámetros:
        F           : campo LightPipes existente
        new_size    : nuevo tamaño físico (en metros)
        wavelength  : longitud de onda del campo
    
    Retorna:
        campo LightPipes con nuevo tamaño físico
    """
    
    # --- Extraer info de LightPipes ---
    A = np.sqrt(Intensity(2, F))     # amplitud
    Phi = Phase(F)                   # fase
    N = A.shape[0]                   # resolución
    
    # --- Normalizar amplitud (evitar saturación numérica) ---
    A = A / np.max(A + 1e-15)
    
    # --- Reinterpolar usando OpenCV ---
    A_rescaled = cv2.resize(A, (N, N), interpolation=cv2.INTER_CUBIC)
    Phi_rescaled = cv2.resize(Phi, (N, N), interpolation=cv2.INTER_CUBIC)

    # --- Crear nuevo campo ---
    F_new = Begin(new_size, wavelength, N)
    F_new = MultIntensity(A_rescaled, F_new)
    F_new = MultPhase(Phi_rescaled, F_new)

    return F_new


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
    Abre un cuadro de diálogo para guardar una imagen a partir de un array numpy.
    """
    # Crear y ocultar ventana de Tkinter
    root = Tk()
    root.withdraw()

    # Cuadro de diálogo "Guardar como"
    ruta_guardado = filedialog.asksaveasfilename(
        title="Guardar imagen",
        defaultextension=".png",
        filetypes=[
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff"),
        ]
    )

    # Si el usuario seleccionó un archivo
    if ruta_guardado:
        # Asegurar que la imagen esté en formato uint8
        if img_array.dtype != np.uint8:
            # Normalizar si es necesario
            img_norm = img_array - img_array.min()
            img_norm = img_norm / (img_norm.max() + 1e-15)
            img_norm = (img_norm * 255).astype(np.uint8)
        else:
            img_norm = img_array

        # Convertir a imagen PIL y guardar
        imagen = Image.fromarray(img_norm)
        imagen.save(ruta_guardado)

    # Destruir raíz oculta
    root.destroy()

