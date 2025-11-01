from optic import ProcesamientoOptico
import matplotlib.pyplot as plt
import numpy as np


#longitud de onda y tamaño de píxel para las pruebas
lambda_test = 633e-9  # 633 nm
pixel_size_test = 10e-6  # 10 micrómetros

#instanciamos la clase ProcesamientoOptico
optical_processor = ProcesamientoOptico(length_wave=lambda_test, pixel_size=pixel_size_test)

#cargamos la imagen del objeto
optical_processor.import_transmitance((1024,1024),255)
#mostramos la transmitancia
optical_processor.show_transmitance()
#z de propagacion
z=0.8 #1 cm
optical_processor.propagate(z)
#mostramos el campo
optical_processor.show_propagated_field()

#ahora pasamos el campo atravez de una lente de foco
f=0.2
optical_processor.lens(f)

#ahora propagamos el campo hasta el foco
optical_processor.propagate(4/15)

#mostramos el campo
optical_processor.show_propagated_field()


image=optical_processor.get_field_propagated()
fig=plt.figure(figsize=(10,12))
plt.imshow(np.angle(image),cmap='gray')
plt.show()
