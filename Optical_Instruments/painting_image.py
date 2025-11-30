from diffraction.utilopctic import export_image
import numpy as np
import matplotlib.pyplot as plt


imagen=np.ones((720,720))

imagen[180:360,180:360]=1
imagen[180:360,360:540]=0.75
imagen[360:540,180:360]=0.5
imagen[360:540,360:540]=0
imagen=imagen

fig=plt.figure(figsize=(6,6))
plt.imshow(imagen, cmap='gray', origin='upper')
plt.axis('off')
plt.title('Imagen de prueba')
plt.show()
export_image(imagen)






