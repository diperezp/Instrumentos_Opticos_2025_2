import numpy as np
import matplotlib.pyplot as plt
from utilopctic import *


Intensidad=import_image()
Phase=import_image()


fig=plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(Intensidad, cmap='gray')
ax1.set_title('Intensidad')
ax1.axis('off')
ax1.grid(False)
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(Phase, cmap='gray')
ax2.set_title('Fase (radianes)')
ax2.axis('off')
ax2.grid(False)
plt.show()