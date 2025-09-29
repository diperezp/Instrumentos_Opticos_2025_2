#%%
import numpy as np
import scipy as sp
from scipy.fft import fft2, ifft2, fftshift, fftfreq
import imageio
import cv2
import scienceplots
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from matplotlib.animation import PillowWriter
import pint
import matplotlib.gridspec as gridspec

plt.style.use(['science', 'notebook'])
u = pint.UnitRegistry()

np.array([1,2]) * u.mm # para hacer los cálculos con unidades automáticamente

# Se genera una funcion que genere el estimulo de entrada, osea la imagen o apertura en terminos opticos 
def generador_u0(imagen,N, dx):
    '''
    Genera el campo complejo de entrada u0 a partir de una imagen.
    
    Parámetros:
    imagen : str
        Ruta de la imagen a cargar.
    dx : float
        Tamaño del pixel en metros.
        
    Retorna:
    u0 : ndarray
        Campo complejo de entrada.
    '''
    # Cargar la imagen en escala de grises
    img = imageio.v2.imread(imagen, mode='F')
    img = np.pad(img, 200, mode='constant')  # Padding para evitar efectos de borde
    img = cv2.resize(img, dsize=(N, N), interpolation=cv2.INTER_CUBIC) # Redimensionar la imagen a 1600x1600 píxeles
    
    # Normalizar la imagen para que los valores estén entre 0 y 1
    img = img / np.max(img)
    
    # Convertir la imagen a un campo complejo (amplitud + fase)
    u0 = img * np.exp(1j * 0)  # Asumiendo fase cero inicialmente
    
    return u0

#-----------------------------------------------------------------------------
#               Parámetros de la simulación
#-----------------------------------------------------------------------------
longitud_onda = 633e-9  # 633 nm
k = 2 * np.pi / longitud_onda
N = 3084  # más puntos → más resolución
L = 1e-2  # tamaño total de la ventana en metros (5 mm)
dx = L / N  # tamaño de píxel en metros
z = 0.40  # 31 mm
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#             1.  Selección de la imagen y generación del campo complejo de entrada
#-----------------------------------------------------------------------------

# abrir un cuadro de diálogo para seleccionar la imagen
Tk().withdraw()  # evita que aparezca la ventana principal de Tkinter
M = filedialog.askopenfilename(
    title="Selecciona una imagen",
    filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
)
u0 = generador_u0(M,N, dx)

# difincion de la rejilla espacial (malla de cordenadas)
x = np.arange(-N/2, N/2) * dx
y = np.arange(-N/2, N/2) * dx
X, Y = np.meshgrid(x, y)

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#               2. Crear U1 que sera la fase esferica de entrada
#-----------------------------------------------------------------------------

#Aqui preraramos la fase esférica que se añade al campo complejo de entrada multiplicando U0 por un Kernel de fase esférica
Kernel_fase_esferica = np.exp(1j * k / (2 * z) * (X**2 + Y**2))
u1 = u0 * Kernel_fase_esferica

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#               3. Transformada de Fresnel usando FFT
#-----------------------------------------------------------------------------

# Transformada de Fresnel usando FFT
U2 = fftshift(fft2(fftshift(u1))) * (dx**2)  # Transformada de Fourier del campo con fase esférica
#fx = fftshift(fftfreq(N, d=dx))  # Frecuencias espaciales en x
#fy = fftshift(fftfreq(N, d=dx))  # Frecuencias espaciales en y
#FX, FY = np.meshgrid(fx, fy)

#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#               4. Escalar el resultado de la Transformada de Fresnel
#-----------------------------------------------------------------------------

# Escalar el resultado de la Transformada de Fresnel
Kernel_escalamiento = (np.exp(1j * k * z) / (1j * longitud_onda * z)) * np.exp(1j * (np.pi * longitud_onda * z) * (X**2 + Y**2) / (longitud_onda * z)**2)
u3 = U2 * Kernel_escalamiento


#-----------------------------------------------------------------------------
#              Visulizacion de todo los campos
#-----------------------------------------------------------------------------

# Definir ejes en micrómetros para graficar
x_um = x * 1e3
y_um = y * 1e3

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # figura más grande

# Campo complejo de entrada
im0 = axs[0].imshow(np.abs(u0), cmap='gray',
                    extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]])
axs[0].set_title('Campo de entrada |u₀|', fontsize=14)
axs[0].set_xlabel("x [mm]")
axs[0].set_ylabel("y [mm]")
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='Amplitud')

# Campo complejo con fase esférica
im1 = axs[1].imshow(np.abs(u1), cmap='plasma',
                    extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]])
axs[1].set_title('Campo con fase esférica |u₁|', fontsize=14)
axs[1].set_xlabel("x [mm]")
axs[1].set_ylabel("y [mm]")
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='Amplitud')

# Campo complejo después de Fresnel
im2 = axs[2].imshow(np.abs(u3), cmap='inferno',
                    extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]])
axs[2].set_title('Difracción de Fresnel |u₃|', fontsize=14)
axs[2].set_xlabel("x [mm]")
axs[2].set_ylabel("y [mm]")
#axs[2].set_xlim(-200, 200)  # Ajustar límites x para mejor visualización
#axs[2].set_ylim(-200, 200)  # Ajustar límites y para mejor visualización
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label='Amplitud')

plt.tight_layout()
plt.show()

# Campo complejo después de Fresnel pero solo
plt.figure(figsize=(18, 6)) 
plt.imshow(np.abs(u3), cmap='inferno', extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]])
plt.title('Difracción de Fresnel |u₃|', fontsize=16)
plt.xlabel("x [mm]", fontsize=14)
plt.ylabel("y [mm]", fontsize=14)
#plt.xlim(-1000, 1000)  # Ajustar límites x para mejor visualización
#plt.ylim(-1000, 1000)  # Ajustar límites y para mejor visualización
plt.colorbar(label='Amplitud')
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------

# %%
