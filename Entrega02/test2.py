from LightPipes import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from optic import import_image_example

# ---------------- Parámetros físicos/simulación ----------------
wavelength = 633*nm      # longitud de onda (m)
N          = 1024        # muestras por lado (N x N)
size       = 5*mm        # tamaño físico del cuadro en el plano objeto (m)
f          = 500*mm      # distancia focal de la lente (m)

# (opcional) diámetro claro de la lente, para limitar NA y evitar aliasing en bordes
lens_radius = 2*mm      # si no quieres limitar, pon None

# ---------------- Campo inicial y transmitancia ----------------
# Campo plano en el plano objeto
F = Begin(size, wavelength, N)

# Cargar imagen como transmitancia de AMPLITUD (0=bloquea, 1=transmite)
img= import_image_example(N)
#intercambia 1 y 0 si quieres que lo oscuro transmita y lo claro bloquee
img =img

amp = np.asarray(img, dtype=np.float64)/255.0


# Aplica transmitancia (intensidad = amp^2). Si prefieres directamente amplitud:
#   F = SubIntensity(F, amp**2)  # fija la intensidad exactamente
F = MultIntensity(F, amp**2)     # multiplica la intensidad del campo actual

# (opcional) limita el haz a un disco en el objeto, si corresponde a tu montaje
# F = CircAperture(F, 2.0*mm)

# ---------------- Propagación: objeto -> lente (f), lente, lente -> sensor (f) ----------------
F = Forvard(F, f)            # propaga hasta el plano de la lente
#tamaño de la lente
lens_radius= 100*mm/2
F = CircAperture(F, lens_radius)  # apertura de la lente (borde físico)
F = Lens(F, f)               # fase de lente delgada 
F= Forvard(F, f)         # propaga distancia extra si es necesario


# ---------------- Intensidad y ejes físicos en el sensor ----------------
I = Intensity(F, flag=0)     # intensidad normalizada
I=np.log(I)

# Escala espacial en el plano sensor (Fourier de la lente):
dx_sensor = wavelength * f / size     # tamaño de pixel en el sensor (m)
L_sensor  = N * dx_sensor             # FOV total (m)
x = (np.arange(N) - N//2) * dx_sensor # eje x (m)
y = (np.arange(N) - N//2) * dx_sensor # eje y (m)

# ---------------- Gráfica ----------------
extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]  # mm

plt.figure(figsize=(6,5))
plt.imshow(I, origin='lower', extent=extent_mm, cmap='gray')
plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
plt.title('Plano sensor (f): |FT{transmitancia}|^2')
plt.colorbar(label='Intensidad (norm.)')
plt.tight_layout()
plt.show()