import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import imageio
import cv2
from tkinter import Tk, filedialog




walength = 633e-9  # Longitud de onda [m]
k = 2 * np.pi / walength  # Número de onda [1/m]
length_side= 5.8e-3 #tamaño de la imagen [m]
N=1080  # Número de pixeles
pixel_size = length_side/N   # Tamaño del pixel [m]

z_max=N*(pixel_size**2)/walength
print(f"z_max: {z_max*1e2:.2f} cm")

z = 0.05# Distancia de propagación [m]

# Crear instancia de la clase AngularSpectrum
#asys=AngularSpectrum()

# Se genera una funcion que genere el estimulo de entrada, osea la imagen o apertura en terminos opticos 
def generador_u0(imagen, dx,N=1024):
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
    img = np.pad(img, 200, mode='constant',constant_values=255)  # Padding para evitar efectos de borde
    img = cv2.resize(img, dsize=(N,N), interpolation=cv2.INTER_CUBIC) # Redimensionar la imagen a 1600x1600 píxeles 
    
    # Normalizar la imagen para que los valores estén entre 0 y 1
    print(img[270,270])
    img = img / np.max(img)
    
    
    # Convertir la imagen a un campo complejo (amplitud + fase)
    u0 = img * np.exp(1j * 0)  # Asumiendo fase cero inicialmente

    
    return u0




#coordenadas de la imagen
x_img = np.linspace(-length_side/2, length_side/2, N)  # Coordenadas x [m]
y_img = np.linspace(-length_side/2, length_side/2, N)   # Coordenadas y [m]
X_img, Y_img = np.meshgrid(x_img, y_img)  # Malla



# Seleccionar archivo de imagen
# abrir un cuadro de diálogo para seleccionar la imagen
Tk().withdraw()  # evita que aparezca la ventana principal de Tkinter
M = filedialog.askopenfilename(
    title="Selecciona una imagen",
    filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
)
print(M)
U0 = generador_u0(M, pixel_size,N)
print(U0.shape)
print(U0.dtype)

#hallamos las coordenadas en el plano de Fourier
fx = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size))
fy = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size))
FX, FY = np.meshgrid(fx, fy)
print(fx.max(), fx.min())

#realizamos la transformada de fourier de la apertura
U0_ft = fftshift(fft2(ifftshift(U0)))
magnitude_spectrum = 20 * np.log(1 + np.abs(U0_ft))

#propagamos el campo
kz = k*np.sqrt((1 - (walength**2)*((FX) ** 2 + (FY) ** 2)).astype(complex))
#H = np.exp(1j * kz * z)  # Función de transferencia


# Opción: elegir si queremos solo componentes propagantes:
only_propagating = True  # <- cambia a True si quieres filtrar evanescentes

if only_propagating:
    prop_mask = ((2*np.pi*FX)**2 + (2*np.pi*FY)**2) <= k**2
    H = np.zeros_like(kz, dtype=complex)
    # sqrt para mask (ya es seguro porque mask selecciona positivos)
    kz_prop = np.sqrt(k**2 - (2*np.pi*FX[prop_mask])**2 - (2*np.pi*FY[prop_mask])**2)
    H[prop_mask] = np.exp(1j * kz_prop * z)
else:

    H = np.exp(1j * kz * z)

    # opcional: truncar componentes con decaimiento numéricamente irrelevante
    alpha = np.maximum(0, np.imag(kz))  # alpha >= 0 for evanescent parts
    too_small = np.exp(-alpha * z) < 1e-12
    H[too_small] = 0





U1 = U0_ft * H
# Campo en el plano de Fourier después de la propagación
intensity = np.log(1+np.abs(U1))
# sacamos la ifft para ver el campo en el espacio
U1_space = fftshift(ifft2(ifftshift(U1)))   



plt.figure(figsize=(12, 6))
# Apertura inicial
plt.subplot(1,5,1)
plt.imshow(np.abs(U0), cmap='gray',extent=[x_img[0]*0.25*1e6, x_img[-1]*0.25*1e6, y_img[0]*0.25*1e6, y_img[-1]*0.25*1e6])
plt.title(f"Apertura inicial\n{length_side*1e6:.0f} µm x {length_side*1e6:.0f} µm")
plt.xlabel("x [µm]")
plt.ylabel("y [µm]")

# Espectro de magnitud
plt.subplot(1,5,2)
plt.imshow(magnitude_spectrum, cmap='gray',extent=[fx[0]*1e-6, fx[-1]*1e-6, fy[0]*1e-6, fy[-1]*1e-6])
plt.title("Espectro de Magnitud")
plt.xlabel("f_x [1/µm]")
plt.ylabel("f_y [1/µm]")

#funcion de transferencia
plt.subplot(1,5,3)
plt.imshow(np.abs(H), cmap='gray',extent=[fx[0]*1e-6, fx[-1]*1e-6, fy[0]*1e-6, fy[-1]*1e-6])
plt.title(f"Función de transferencia\nz={z*1e2:.1f} cm")
plt.xlabel("f_x [1/µm]")
plt.ylabel("f_y [1/µm]")    

# Intensidad después de la propagación
plt.subplot(1,5,4)
plt.imshow(intensity, cmap='gray',extent=[fx[0]*1e-6, fx[-1]*1e-6, fy[0]*1e-6, fy[-1]*1e-6])
plt.title(f"propagacion en el dominio espectral")
plt.xlabel("x [1/µm]")
plt.ylabel("y [1/µm]")

# Intensidad en el espacio después de la propagación
plt.subplot(1,5,5)
plt.imshow(np.log(1+np.abs(U1_space)), cmap='gray',extent=[0.25*x_img[0]*1e6, 0.25*x_img[-1]*1e6, 0.25*y_img[0]*1e6, 0.25*y_img[-1]*1e6])
plt.title(f"Campo en el espacio\nz={z*1e2:.1f} cm")
plt.xlabel("x [µm]")
plt.ylabel("y [µm]")


plt.tight_layout()
plt.show()

#imprimimos la informacion del arreglo de la apertura
print(f"Arreglo de la apertura: {U0.shape}")
print(f"Valor máximo de la apertura: {U0.max()}")
print(f"Valor mínimo de la apertura: {U0.min()}")
print(f"Tipo de dato de la apertura: {U0.dtype}")
print(f"Valor del pixel central: {U0[N//2, N//2]}")
print(f"Valor del pixel en (0,0): {U0[0, 0]}")
print(f"Valor del pixel en (N-1,N-1): {U0[N-1, N-1]}")
print(U0)