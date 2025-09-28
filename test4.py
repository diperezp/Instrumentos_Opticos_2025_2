#importamos la libreria backpropagation_angular_espectrum
from backpropagation_angular_espectrum import *
from optic_util import import_image as import_1
from optic_util import *



img = import_1(None)  # Cargar imagen de ejemplo
plot_image(img, title='Imagen Original', cmap_type='gray')

#instanciamos la clase metodo_gerchberg_saxton
metodo = metodo_gerchberg_saxton(img,20)
#reconstruimos la fase
reconstructed_field = metodo.reconstruct()
#mostramos la imagen reconstruida
print(reconstructed_field.shape)
plot_image(reconstructed_field, title='Imagen Reconstruida', cmap_type='gray')
