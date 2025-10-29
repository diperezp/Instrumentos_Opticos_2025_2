from utilopctic import *
from scale import *

image=import_image()[:,:,0]  #cargamos solo un canal
print("image.shape:", image.shape)
X,Y=mesh_image(image,0.001,0.001)
plot_image(image,X,Y)

resized_image=resize_with_pad(image,(1024,1024))
print("resized_image.shape:", resized_image.shape)
X_resized,Y_resized=mesh_image(resized_image,0.001*2,0.001*2)
plot_image(resized_image,X_resized,Y_resized)   