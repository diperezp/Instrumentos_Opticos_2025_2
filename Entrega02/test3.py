from LightPipes import *
from matplotlib import pyplot as plt
import numpy as np
from optic import import_image_example

wavelength = 633*nm      # longitud de onda (m)
size       = 7*mm        # tamaño físico del cuadro en el plano objeto (m)
N= 1024        # muestras por lado (N x N)
w0=1*mm        # radio de la waists del haz gaussiano (m)
f=1*m      # distancia focal de la lente (m)
z=0.5*m          # distancia de propagación adicional (m)

M_lens=[[0.5,0],[ -1/f,2]]  # matriz ABCD de la lente delgada

M_propag=[[1,z],[0,1]]  # matriz ABCD de la propagación libre de distancia z

F=Begin(size,wavelength,N)  # crea el campo inicial
# Cargar imagen como transmitancia de AMPLITUD (0=bloquea, 1=transmite)
img= import_image_example(N)
#intercambia 1 y 0 si quieres que lo oscuro transmita y lo claro bloquee
img =img
amp = np.asarray(img, dtype=np.float64)/255.0
F = MultIntensity(F, amp**2)     # multiplica la intensidad del campo actual
F=ABCD(F,M_lens)          # pasa por la lente delgada
print("Después de la lente:")
F=ABCD(F,M_propag)        # propaga distancia z


# ---------------- Intensidad y ejes físicos en el sensor ----------------
I = Intensity(F, flag=1)     # intensidad normalizada

# Escala espacial en el plano sensor (Fourier de la lente):
dx_sensor = wavelength * f / size     # tamaño de pixel en el sensor (m)
L_sensor  = N * dx_sensor             # FOV total (m)
x = (np.arange(N) - N//2) * dx_sensor # eje x (m)
y = (np.arange(N) - N//2) * dx_sensor # eje y (m)

# ---------------- Gráfica ----------------
extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]  # mm

plt.figure(figsize=(6,5))
plt.imshow(I, origin='lower', extent=extent_mm, cmap='inferno')
plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
plt.title('Plano sensor (f): |FT{transmitancia}|^2')
plt.colorbar(label='Intensidad (norm.)')
plt.tight_layout()
plt.show()
