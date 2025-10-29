import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import cv2

def mesh_image(image, delta_x, delta_y):
    """
    Esta funcion devulve la grilla relacionada """
   

    rows, cols = image.shape
    x = np.linspace(-delta_x*cols/2, cols * delta_x/2, cols)
    y = np.linspace(-delta_y*rows/2, rows * delta_y/2, rows)
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_image(image,X,Y):
  """Esta funcion grafica la funcion con su malla real

  image:np.array imagen a graficar
  X:np.array malla real en el eje x
  Y:np.array malla real en el eje y
  """
  
  fig=plt.figure(figsize=(10,10))
  plt.imshow(image,extent=[X.min(),X.max(),Y.min(),Y.max()], cmap='gray', origin='upper')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()

def resize_with_pad(image:np.ndarray,new_size:Tuple[int,int],pad_value:int=255)->np.ndarray:
    """
    Redimensionamos la imagen y luego aplicamos padding.
    """
    # Redimensionar la imagen
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Calcular el tamaño del padding necesario
    pad_height = (new_size[0]) // 2
    pad_width = (new_size[1]) // 2
    print("pad_width,pad_height:", pad_width, pad_height)

    # Aplicar padding a la imagen
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=pad_value)

    return padded_image