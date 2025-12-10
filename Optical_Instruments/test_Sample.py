from polarizacion.library.stokes import birrefrigente,polarizador

import matplotlib.pyplot as plt
import numpy as np
import math



#instanciamos la clase birregrigente
sample=polarizador()


Ex_out,Ey_out=sample.apply_initial_polarization(jones_in=[1,0])



#lienzo
fig=plt.figure(figsize=(10,12))
axis1=fig.add_subplot(1,3,1)
axis2=fig.add_subplot(1,3,2)
axis3=fig.add_subplot(1,3,3)



#mostramos el campo E_x
Intensidad_X=np.abs(Ex_out)
axis1.imshow(Intensidad_X,cmap='gray',vmin=0,vmax=1)

#mostramos el campo E_y
Intensidad_Y=np.abs(Ey_out)
axis2.imshow(Intensidad_Y,cmap='gray',vmin=0,vmax=1)

#mostramos la intensidad de ambos
Intensidad_neta=np.abs(Ex_out+Ey_out)
axis3.imshow(Intensidad_neta,cmap='gray',vmin=0,vmax=1)




plt.show()




