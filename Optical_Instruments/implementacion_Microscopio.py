import numpy as np
from PIL import Image

# Tamaño de la imagen
N = 2048

# Crear patrón: 1 pixel blanco / 1 pixel negro
pattern = np.zeros((N, N), dtype=np.uint8)

for x in range(N):
    if x % 2 == 0:
        pattern[:, x] = 255  # blanco
    else:
        pattern[:, x] = 0    # negro

# Guardar imagen
img = Image.fromarray(pattern, mode='L')
img.save("patron_lineas_1px_2048.png")
