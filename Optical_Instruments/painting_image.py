from diffraction.utilopctic import export_image
import numpy as np
import matplotlib.pyplot as plt


imagen=np.zeros((720,720))

imagen[0:360,0:360]=1
imagen[360:720,360:720]=0.5
imagen[0:360,360:720]=0
imagen[360:720,0:360]=0.75
imagen=1-imagen

fig=plt.figure(figsize=(6,6))
plt.imshow(imagen, cmap='gray', origin='upper')
plt.axis('off')
plt.title('Imagen de prueba')
plt.show()
export_image(imagen)






