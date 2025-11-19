import numpy as np

# Genera números aleatorios 
data = np.random.randint(0, 201, size=(10,10))

# Muestra los números aleatorios 
print(data)

# --- Funciones que reciben el arreglo ---

def metodoPoker(arreglo):
    # Aquí podrás trabajar con el arreglo
    print("Accediendo al arreglo desde metodoPoker:")
    print(arreglo)


def metodoHuecos(arreglo):
    # Aquí podrás trabajar con el arreglo
    print("Accediendo al arreglo desde metodoHuecos:")
    print(arreglo)


# --- Llamada a las funciones pasándoles el arreglo ---

metodoPoker(data)
metodoHuecos(data)
