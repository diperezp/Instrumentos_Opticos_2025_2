import numpy as np
import matplotlib.pyplot as plt


def mascara_circular(xx: np.ndarray, yy: np.ndarray, radio: float) -> np.ndarray:
    """Máscara binaria circular centrada (1 dentro del disco, 0 fuera)."""
    return ((xx**2 + yy**2) <= radio**2).astype(float)

#características de iluminacion
wavelength = 533e-9  # Longitud de onda en metros
f_MO = 10e-3  # Distancia focal del objetivo en metros


#Caracteristicas de la imagen
N_x = 1024 # Ancho de la imagen
N_y = 1024 # Alto de la imagen
#dimensiones físicas de la imagen en m
size_x = 390e-6  # m
size_y = 390e-6  # m
#tamaño del pixel
dx = size_x / N_x  # Tamaño del pixel en x
dy = size_y / N_y  # Tamaño del pixel en y
#malla de coordenadas
x = np.linspace(-size_x/2, size_x/2, N_x)
y = np.linspace(-size_y/2, size_y/2, N_y)
X, Y = np.meshgrid(x, y)
#tamaño del radio de la máscara circular
R_pupila = 10e-3  # Radio en metros
R_filter = 1.22*wavelength*f_MO/R_pupila

print(f"Radio pupila: {R_pupila*1e3} mm")
print(f"Radio filtro Zernike: {R_filter*1e9} nm")





#creamos a lienzo de Zernike vacío
Pupila = mascara_circular(X, Y, R_pupila)

Pupila = Pupila.astype(np.complex128)

Filter = mascara_circular(X, Y, R_filter)
Pupila =Pupila - Filter
Filter = Filter.astype(np.complex128)
#aplicamos alguna función de Zernike
Filter = Filter*np.exp(1j*np.pi/2)/2

img = Pupila + Filter


#Creamos un patrón de Zernike simple (por ejemplo, un círculo

Intensidad=np.abs(img)

Phase=np.angle(img)



def filter_Zernike(X, Y, R_pupila, R_filter, coef,Beta):
    # Creamos la pupila
    Pupila = mascara_circular(X, Y, R_pupila).astype(np.complex128)
    
    # Creamos el filtro Zernike
    Filter = mascara_circular(X, Y, R_filter).astype(np.complex128)

    Pupila = Pupila - Filter
    
    # Aplicamos el coeficiente de Zernike al filtro
    Filter = Beta*Filter * np.exp(1j * coef)
    
    # Combinamos la pupila y el filtro
    img = Pupila + Filter
    
    return img

Filter = filter_Zernike(X, Y, R_pupila, R_filter, np.pi/2,0.5)


fig=plt.figure(figsize=(10,12))
ax0=fig.add_subplot(221)
im0=ax0.imshow(Intensidad, cmap='gray')
ax0.set_label('Intensidad')
cbar0=plt.colorbar(im0, ax=ax0, label='Intensidad', fraction=0.046, pad=0.04)  # muestra la escala de colores
ax1=fig.add_subplot(222)
im1=ax1.imshow(Phase, cmap='gray')
ax1.set_label('Fase (radianes)')  # etiqueta de la colorbar
cbar1=plt.colorbar(im1, ax=ax1, label='Fase (radianes)', fraction=0.046, pad=0.04)  # muestra la escala de colores
ax2=fig.add_subplot(223)
im2=ax2.imshow(np.abs(Filter), cmap='gray', origin='lower')   # guarda el objeto de imagen
cbar2 = plt.colorbar(im2, ax=ax2, label='Magnitud', fraction=0.046, pad=0.04)  # muestra la escala de colores
ax3=fig.add_subplot(224)
im3=ax3.imshow(np.angle(Filter), cmap='gray', origin='lower')   # guarda el objeto de imagen
cbar3 = plt.colorbar(im3, ax=ax3, label='Fase (radianes)', fraction=0.046, pad=0.04)  # muestra la escala de colores
plt.show()
