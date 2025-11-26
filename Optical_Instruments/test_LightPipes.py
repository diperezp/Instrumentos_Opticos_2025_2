from diffraction.fresnel import Field


wavelength = 500e-9  # Wavelength in meters
grid_size = 1e-3     # Grid size in meters
N = 1024              # Number of grid points


field = Field(grid_size, wavelength, N)
# field.import_Intensity(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/amplitud.png')
# field.import_Phase(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/fase.png')
field.import_Intensity(path=None)

field.show_intensity()

# field.show_intensity()
# field.show_phase()

#calculamos el limite de Fresnel
print("Límite de Fresnel:", field.fresnel_limit())

z=5e-3

field.fresnel_propagation(z)
field.lens(z)
field.pupila(5e-4)
field.fresnel_propagation(z)



field.fresnel_propagation(z)
field.lens(z)
field.pupila(5e-4)
field.fresnel_propagation(z)


field.show_intensity()
field.show_phase()