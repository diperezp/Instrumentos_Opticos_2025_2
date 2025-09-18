import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

walenghth = 500e-9  # Longitud de onda [m]
k = 2 * np.pi / walenghth  # Número de onda [1/m]
pixel_size = 39e-9  # Tamaño del pixel [m]
N=1024  # Número de pixeles
z = 3.125e-7 # Distancia de propagación [m]



#coordenadas de la imagen
x_img = np.linspace(-20e-6, 20e-6, N)  # Coordenadas x [m]
y_img = np.linspace(-20e-6, 20e-6, N)   # Coordenadas y [m]
X_img, Y_img = np.meshgrid(x_img, y_img)  # Malla

# Apertura rectangular
slit_width = 4e-6  # Ancho de la rendija [m]
slit_height = 4e-6  # Alto de la rendija [m]
U0 = np.where((np.abs(X_img) <= slit_width / 2) & (np.abs(Y_img) <= slit_height / 2), 1, 0)

#hallamos las coordenadas en el plano de Fourier
fx = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size))
fy = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size))
FX, FY = np.meshgrid(fx, fy)
print(fx.max(), fx.min())

#realizamos la transformada de fourier de la apertura
U0_ft = fftshift(fft2(ifftshift(U0)))
magnitude_spectrum = 20 * np.log(1 + np.abs(U0_ft))

#propagamos el campo
kz = k*np.sqrt((1 - (walenghth**2)*((FX) ** 2 + (FY) ** 2)).astype(complex))
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
plt.imshow(U0[256:768,256:768], cmap='gray',extent=[x_img[0]*0.25*1e6, x_img[-1]*0.25*1e6, y_img[0]*0.25*1e6, y_img[-1]*0.25*1e6])
plt.title(f"Apertura inicial\n{slit_width*1e6:.0f} µm x {slit_height*1e6:.0f} µm")
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
plt.imshow(np.log(1+np.abs(U1_space[256:768,256:768])), cmap='gray',extent=[0.25*x_img[0]*1e6, 0.25*x_img[-1]*1e6, 0.25*y_img[0]*1e6, 0.25*y_img[-1]*1e6])
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