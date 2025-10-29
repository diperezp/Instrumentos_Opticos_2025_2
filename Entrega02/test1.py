from optic import ProcesamientoOptico
import matplotlib.pyplot as plt


#longitud de onda y tamaño de píxel para las pruebas
lambda_test = 633e-9  # 633 nm
pixel_size_test = 10e-6  # 10 micrómetros

#instanciamos la clase ProcesamientoOptico
optical_processor = ProcesamientoOptico(length_wave=lambda_test, pixel_size=pixel_size_test)

#importamos una imagen del directorio actual
image=optical_processor.import_transmitance(new_shape=(512,512),value_pad=0)

print("Dimensiones de la transmitancia importada:", image.shape)

#mostramos la transmitancia importada
optical_processor.show_transmitance()

#realizamos la propagación de Fresnel a una distancia z=0.1 m
z_propagation = 0.1  # 0.1 metros
optical_processor.propagate(z=z_propagation, mode='fresnel')
#mostramos el campo propagado
optical_processor.show_propagated_field()
#obtenemos el campo propagado
field_propagated = optical_processor.get_field_propagated()
#mostramos el campo propagado con matplotlib para verificar
plt.imshow(np.angle(field_propagated), cmap='gray')
plt.title(  'fase adquiridada (Fre'snel) ')             


image=optical_processor.get_transmitance()
#mostramos image con matplotlib para verificar
plt.imshow(abs(image), cmap='gray')
plt.title('Transmitancia Importada')
plt.colorbar()
plt.show()
