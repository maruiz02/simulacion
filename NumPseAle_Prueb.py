import numpy as np
from collections import Counter
import math

# ========================================
#   CLASIFICACIÓN DE MANO PÓKER
# ========================================
def clasificar_mano(digitos):
    c = Counter(digitos)
    rep = sorted(c.values(), reverse=True)

    if rep == [5]:
        return "Quintilla"
    if rep == [4, 1]:
        return "Poker"
    if rep == [3, 2]:
        return "Full House"
    if rep == [3, 1, 1]:
        return "Tercia"
    if rep == [2, 2, 1]:
        return "Dos Pares"
    if rep == [2, 1, 1, 1]:
        return "Un Par"
    if rep == [1, 1, 1, 1, 1]:
        return "Todos Diferentes"
    return "Error"

# ========================================
#   PRUEBA DE PÓKER
# ========================================
def metodoPoker(arreglo):

    probabilidades = {
        "Todos Diferentes": 0.3024,
        "Un Par": 0.5040,
        "Dos Pares": 0.1080,
        "Tercia": 0.0720,
        "Full House": 0.0090,
        "Poker": 0.0045,
        "Quintilla": 0.0001
    }

    conteo = {k: 0 for k in probabilidades.keys()}

    for num in arreglo:
        digitos = list(f"{num:.5f}".split(".")[1])
        mano = clasificar_mano(digitos)
        conteo[mano] += 1

    n = len(arreglo)
    esperados = {k: v * n for k, v in probabilidades.items()}

    chi = sum((conteo[k] - esperados[k])**2 / esperados[k] for k in conteo)

    return chi < 12.592  # True si pasa la prueba

# ========================================
#   PRUEBA DE HUECOS
# ========================================
def metodoHuecos(arreglo):

    a_int = 0.1
    b_int = 1.0

    huecos = []
    contador = 0
    dentro = False

    for x in arreglo:
        if a_int <= x <= b_int:
            if dentro:
                huecos.append(contador)
            dentro = True
            contador = 0
        else:
            if dentro:
                contador += 1

    if not huecos:
        return False

    frec_obs = Counter(huecos)

    p = b_int - a_int
    q = 1 - p

    n = sum(frec_obs.values())
    frec_esp = {h: n * (p * (q**h)) for h in frec_obs}

    chi = sum((frec_obs[h] - frec_esp[h])**2 / frec_esp[h] for h in frec_obs)

    return chi < 12.592

# ========================================
#   PRUEBA DE CORRIDAS ARRIBA Y ABAJO
# ========================================
def metodoCorridasAyAb(arreglo):

    signos = []

    for i in range(len(arreglo) - 1):
        if arreglo[i+1] > arreglo[i]:
            signos.append("+")
        elif arreglo[i+1] < arreglo[i]:
            signos.append("-")

    if len(signos) < 2:
        return False

    corridas = 1
    for i in range(1, len(signos)):
        if signos[i] != signos[i - 1]:
            corridas += 1

    N = len(signos)

    Esperanza = (2*N - 1) / 3
    Varianza = (16*N - 29) / 90

    Z = (corridas - Esperanza) / math.sqrt(Varianza)

    return abs(Z) < 1.96

# ========================================
#   CICLO QUE GENERA ARREGLOS HASTA PASAR 2 PRUEBAS
# ========================================

# === NUEVO: pedir tamaño del arreglo ===
tam = int(input("¿De qué tamaño quieres el arreglo?: "))

intentos = 0

while True:
    intentos += 1

    a = np.round(np.random.random(size=tam), 5)

    pasaPoker = metodoPoker(a)
    pasaHuecos = metodoHuecos(a)
    pasaCorridas = metodoCorridasAyAb(a)

    total = pasaPoker + pasaHuecos + pasaCorridas

    print(f"\nIntento {intentos}: Póker={pasaPoker}, Huecos={pasaHuecos}, Corridas={pasaCorridas}")

    if total >= 2:
        print("\n===================================")
        print(" EL ARREGLO FINAL PASÓ AL MENOS 2 PRUEBAS ")
        print("===================================")
        print("\nPruebas aprobadas:")
        if pasaPoker: print(" Póker")
        if pasaHuecos: print(" Huecos")
        if pasaCorridas: print(" Corridas Arriba y Abajo")

        print("\nArreglo utilizado:")
        print(a)

        break
