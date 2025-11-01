import numpy as np
import matplotlib.pyplot as plt
from utilopctic import *
from scipy.fft import fft2, ifft2, fftshift
from LightPipes import *
from utilopctic import *
from scale import *
import cv2

#condiciones fisicas
λ = 632.8e-9
size = 5*mm
N = 4096

#importamos nuestra imagen
img=import_image()[:,:]
#redimensionamos la imagen
#img=cv2.resize(img,(2048,2048),interpolation=cv2.INTER_AREA)
img=resize_with_pad(img,(2048,2048),0)
#normalizamos la imagen



#instanciamos la clase 
U=Begin(size,λ,N,)

#Insertamos la transmitancia
U = MultIntensity(U,img)

I=Intensity(0,U)


fig=plt.figure(figsize=(10,12))
plt.imshow(np.log(1+I),cmap='gray')
plt.show()


#propagamos el campo
f=0.1

U=Forvard(U,0.1)
U=Lens(U,f)
U=Forvard(U,0.1)
#U=GaussAperture(U,10*um,10*um,10*um)

U=CircScreen(U,70*um,400*um,-1.400*mm)
U=CircScreen(U,70*um,0.85*mm,-0.85*mm)
U=CircScreen(U,70*um,2.10*mm,-0.7*mm)
U=CircScreen(U,70*um,-400*um,1.400*mm)
U=CircScreen(U,70*um,-0.85*mm,0.85*mm)
U=CircScreen(U,70*um,-2.10*mm,0.7*mm)





I=Intensity(0,U)


fig=plt.figure(figsize=(10,12))
plt.imshow(np.log(1+I),cmap='gray')
plt.show()

U=Forvard(U,f)
U=Lens(U,f)
U=Forvard(f,U)

I=Intensity(0,U)


fig=plt.figure(figsize=(10,12))
plt.imshow(np.log(1+I),cmap='gray')
plt.show()


