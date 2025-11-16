from optic import ProcesamientoOptico
import matplotlib.pyplot as plt
import numpy as np


#longitud de onda y tamaño de píxel para las pruebas
lambda_test = 633e-9  # 633 nm
pixel_size_test = 10e-6  # 10 micrómetros

#instanciamos la clase ProcesamientoOptico
optical_processor = ProcesamientoOptico(length_wave=lambda_test, pixel_size=pixel_size_test)

#importamos una imagen del directorio actual
image=optical_processor.import_transmitance(878)

print("Dimensiones de la transmitancia importada:", image.shape)

#mostramos la transmitancia importada
optical_processor.show_transmitance()

#realizamos la propagación de Fresnel a una distancia z=0.1 m
z_propagation = 0.1  # 0.1 metros
optical_processor.propagate(z=z_propagation, mode='fresnel')
optical_processor.show_propagated_field()



#mostramos image con matplotlib para verificar
plt.imshow(np.log(1+np.abs(image)), cmap='gray')
plt.title('Transmitancia Importada')
plt.colorbar()
plt.show()
