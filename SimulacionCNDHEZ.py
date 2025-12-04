# ==============================
# IMPORTACIÓN DE LIBRERÍAS
# ==============================
import simpy
import numpy as np
import pandas as pd
from collections import Counter
import math
import matplotlib.pyplot as plt 
import seaborn as sns 


# ======================================================================
# 0. VALIDACIÓN DEL ARREGLO DE NÚMEROS ALEATORIOS
# ======================================================================

def clasificar_mano(digitos):
    c = Counter(digitos)
    rep = sorted(c.values(), reverse=True)

    if rep == [5]: return "Quintilla"
    if rep == [4, 1]: return "Poker"
    if rep == [3, 2]: return "Full House"
    if rep == [3, 1, 1]: return "Tercia"
    if rep == [2, 2, 1]: return "Dos Pares"
    if rep == [2, 1, 1, 1]: return "Un Par"
    if rep == [1, 1, 1, 1, 1]: return "Todos Diferentes"
    return "Error"


def metodoPoker(arreglo):
    probabilidades = {
        "Todos Diferentes": 0.3024, "Un Par": 0.5040, "Dos Pares": 0.1080, "Tercia": 0.0720, 
        "Full House": 0.0090, "Poker": 0.0045, "Quintilla": 0.0001
    }
    conteo = {k: 0 for k in probabilidades.keys()}
    for num in arreglo:
        digitos = list(f"{num:.5f}".split(".")[1])
        mano = clasificar_mano(digitos)
        conteo[mano] += 1
    n = len(arreglo)
    esperados = {k: v * n for k, v in probabilidades.items()}
    chi = sum((conteo[k] - esperados[k])**2 / esperados[k] for k in conteo if esperados[k] > 0)
    return chi < 12.592


def metodoHuecos(arreglo):
    a_int = 0.1
    b_int = 1.0
    huecos = []
    contador = 0
    dentro = False
    for x in arreglo:
        if a_int <= x <= b_int:
            if dentro: huecos.append(contador)
            dentro = True 
            contador = 0
        elif dentro: contador += 1
    if not huecos: return False
    frec_obs = Counter(huecos)
    p = b_int - a_int
    q = 1 - p
    n = sum(frec_obs.values())
    frec_esp = {h: n * (p * (q**h)) for h in frec_obs}
    chi = sum((frec_obs[h] - frec_esp[h])**2 / frec_esp[h] for h in frec_obs if frec_esp[h] > 0)
    return chi < 12.592


def metodoCorridasAyAb(arreglo):
    signos = []
    for i in range(len(arreglo) - 1):
        if arreglo[i+1] > arreglo[i]: signos.append("+")
        elif arreglo[i+1] < arreglo[i]: signos.append("-")
    if len(signos) < 2: return False
    corridas = 1
    for i in range(1, len(signos)):
        if signos[i] != signos[i - 1]: corridas += 1
    N = len(signos)
    Esperanza = (2*N - 1) / 3
    Varianza = (16*N - 29) / 90
    if Varianza <= 0: return False
    Z = (corridas - Esperanza) / math.sqrt(Varianza)
    return abs(Z) < 1.96


# ================================================================
# CICLO QUE GENERA ARREGLOS HASTA PASAR 2 PRUEBAS
# ================================================================
TAM_MUESTRA = 2000
SEED_VALIDADA = None
intentos = 0

while True:
    intentos += 1
    a = np.round(np.random.random(size=TAM_MUESTRA), 5)
    
    if metodoPoker(a) + metodoHuecos(a) + metodoCorridasAyAb(a) >= 2:
        SEED_VALIDADA = int(a[0] * 100000)
        np.random.seed(SEED_VALIDADA)
        print(f"Semilla validada: {SEED_VALIDADA}")
        break

    if intentos > 100:
        print("No se encontró una semilla validada. Usando valor por defecto.")
        SEED_VALIDADA = 12345
        np.random.seed(SEED_VALIDADA)
        break


# ======================================================================
# 1. PARÁMETROS Y CONFIGURACIÓN
# ======================================================================

NUM_REPLICAS = 30
DURACION_SIMULACION = 11 * 365.25
DIAS_POR_ANIO = 365.25

NUM_VISTADURIAS = 11
MEDIA_QUEJAS_TRIMESTRAL = 165
DS_QUEJAS_TRIMESTRAL = 241


# =======================================================
# CATEGORÍAS + AGREGADA "OTROS"
# =======================================================

CATEGORIAS = [
    {"nombre": "Incompetencia", "prob": 0.03322, "tipo": "triangular", "tiempos": [5, 7, 12]},
    {"nombre": "Desistimiento", "prob": 0.17311, "tipo": "exponencial", "tiempos": [10]},
    {"nombre": "Falta de interes", "prob": 0.08824, "tipo": "uniforme", "tiempos": [4, 9]},
    {"nombre": "Conciliaciones", "prob": 0.05398, "tipo": "triangular", "tiempos": [3, 4, 10]},
    {"nombre": "Solucionados en tramite", "prob": 0.23203, "tipo": "exponencial", "tiempos": [15]},
    {"nombre": "Acuerdo de NO responsabilidad", "prob": 0.15702, "tipo": "uniforme", "tiempos": [6, 8]},
    {"nombre": "Recomendaciones", "prob": 0.05139, "tipo": "exponencial", "tiempos": [11]},
    {"nombre": "Improcedencias", "prob": 0.09473, "tipo": "triangular", "tiempos": [5, 6, 20]},
    {"nombre": "Allaneamiento", "prob": 0.01479, "tipo": "uniforme", "tiempos": [3, 5]},
    {"nombre": "Falta de Materia", "prob": 0.00934, "tipo": "exponencial", "tiempos": [7]},
    {"nombre": "Otros", "prob": 0.09215, "tipo": "uniforme", "tiempos": [5, 10]}
]

# VERIFICAR SUMA = 1
assert abs(sum(c["prob"] for c in CATEGORIAS) - 1.0) < 0.0001, "Las probabilidades NO suman 1"


# =======================================================
# Selección dinámica de categoría
# =======================================================
def seleccionar_categoria():
    r = np.random.rand()
    suma = 0
    for cat in CATEGORIAS:
        suma += cat["prob"]
        if r < suma:
            return cat
    return CATEGORIAS[-1]


# ======================================================================
# 2. PROCESO DE LA QUEJA (Entidad)
# ======================================================================

def proceso_queja_v2(env, nombre_queja, recursos, resultados_list):

    tiempo_llegada = env.now
    cat = seleccionar_categoria()

    if cat["tipo"] == "triangular":
        tiempo_servicio = np.random.triangular(*cat["tiempos"])
    elif cat["tipo"] == "exponencial":
        tiempo_servicio = np.random.exponential(cat["tiempos"][0])
    else:
        tiempo_servicio = np.random.uniform(*cat["tiempos"])

    tiempo_servicio = max(0.1, tiempo_servicio)

    with recursos.request() as req:
        yield req

        tiempo_espera = env.now - tiempo_llegada
        yield env.timeout(tiempo_servicio)

        tiempo_res = env.now
        tiempo_total = tiempo_res - tiempo_llegada

        rand_anual = np.random.rand()
        res_anio = (rand_anual >= 0.51) and (rand_anual <= 0.99)

        anio_res = math.floor(tiempo_res / DIAS_POR_ANIO) + 1

        resultados_list.append({
            'ID': nombre_queja,
            'Categoria': cat["nombre"],
            'TiempoLlegada': tiempo_llegada,
            'TiempoResolucion': tiempo_res,
            'TiempoEspera': tiempo_espera,
            'TiempoTotal': tiempo_total,
            'ResueltaEnAnio': res_anio,
            'Año': anio_res
        })


# ======================================================================
# GENERADOR DE ARRIBOS
# ======================================================================

def generador_arribos_v2(env, recursos, resultados_list):

    queja_id = 0

    while True:
        yield env.timeout(90)

        cant = int(np.random.normal(MEDIA_QUEJAS_TRIMESTRAL, DS_QUEJAS_TRIMESTRAL))
        cant = max(0, cant)

        for _ in range(cant):
            queja_id += 1
            env.process(proceso_queja_v2(env, queja_id, recursos, resultados_list))


# ======================================================================
# FUNCIÓN DE REPLICA
# ======================================================================

all_results_dfs = []
metricas_replicas = []

def run_replica(replica_num):

    resultados_replica_actual = [] 

    env = simpy.Environment()
    recursos = simpy.Resource(env, capacity=NUM_VISTADURIAS)

    env.process(generador_arribos_v2(env, recursos, resultados_replica_actual))
    env.run(until=DURACION_SIMULACION)

    if resultados_replica_actual:
        df = pd.DataFrame(resultados_replica_actual)

        metricas = {
            'Replica': replica_num,
            'Total_Resueltas': df.shape[0],
            'TiempoTotal_Promedio': df['TiempoTotal'].mean(),
            'TiempoEspera_Promedio': df['TiempoEspera'].mean(),
            'Prob_ResueltaEnAnio': df['ResueltaEnAnio'].mean()
        }

        df["Replica"] = replica_num
        all_results_dfs.append(df)
        metricas_replicas.append(metricas)

        print(f"Replica {replica_num} lista ({df.shape[0]} quejas)")


# ======================================================================
# EJECUCIÓN DE LAS RÉPLICAS
# ======================================================================
print("\n" + "═"*50)
print(f"INICIANDO SIMULACIÓN DE {NUM_REPLICAS} RÉPLICAS")
print("═"*50)

for i in range(1, NUM_REPLICAS + 1):
    np.random.seed(SEED_VALIDADA + i)
    run_replica(i)


# ======================================================================
# 5. ANÁLISIS AGREGADO
# ======================================================================

if metricas_replicas:

    df_metricas = pd.DataFrame(metricas_replicas)
    print("\n" + "█"*50)
    print("MÉTRICAS AGREGADAS (PROMEDIO DE RÉPLICAS)")
    print("█"*50)
    
    metricas_finales = df_metricas[['Total_Resueltas', 'TiempoTotal_Promedio', 
                                    'TiempoEspera_Promedio', 'Prob_ResueltaEnAnio']].agg(['mean', 'std'])
    print(metricas_finales.T.rename(columns={'mean': 'Media', 'std': 'Desviación Estándar'}).round(2))

    df_consol = pd.concat(all_results_dfs, ignore_index=True)

    df_prom_anual = df_consol.groupby(["Replica", "Año"]).size().reset_index(name="Total")
    df_prom_anual_media = df_prom_anual.groupby("Año")["Total"].mean().reset_index(name="Quejas_Media")
    df_prom_anual_media['Año'] = df_prom_anual_media['Año'].astype(int)

    print("\nQuejas resueltas promedio por año:\n")
    print(df_prom_anual_media.round(0))


    # ======================================================================
    # 6. VISUALIZACIÓN DE RESULTADOS AGREGADOS 
    # ======================================================================
    
    sns.set_style("whitegrid")
    
    ### GRÁFICO 1: Quejas Resueltas por Año (Promedio)
    plt.figure(figsize=(10, 5))
    # 🟢 CORRECCIÓN DE WARNING: Añadido hue='Año' y legend=False
    sns.barplot(x='Año', y='Quejas_Media', data=df_prom_anual_media, hue='Año', palette='viridis', legend=False)
    plt.title(f'Quejas Resueltas por Año (Promedio de {NUM_REPLICAS} Réplicas)', fontsize=16) 
    plt.xlabel('Año de la Simulación')
    plt.ylabel('Media de Quejas Resueltas')
    plt.xticks(rotation=0)
    for index, row in df_prom_anual_media.iterrows():
        plt.text(row.name, row.Quejas_Media, f'{row.Quejas_Media:.0f}', color='black', ha="center", va="bottom")
    plt.tight_layout()
    plt.show() 
    
    
    ### GRÁFICO 2: Distribución del Tiempo Total Promedio (Histograma de Réplicas)
    plt.figure(figsize=(10, 5))
    sns.histplot(df_metricas['TiempoTotal_Promedio'], bins=10, kde=True, color='darkorange')
    
    media_total = df_metricas['TiempoTotal_Promedio'].mean()
    plt.axvline(media_total, color='red', linestyle='--', label=f'Media Global: {media_total:.2f} días')
    
    plt.title(f'Distribución del Tiempo Total Promedio (sobre {NUM_REPLICAS} Réplicas)', fontsize=16) 
    plt.xlabel('Tiempo Total Promedio (días)')
    plt.ylabel('Frecuencia de Réplicas')
    plt.legend()
    plt.tight_layout()
    plt.show() 
    
    
    ### GRÁFICO 3: Quejas Resueltas por Categoría (Promedio de Réplicas)
    
    df_prom_categoria = df_consol.groupby('Categoria').size().reset_index(name='Total_Resueltas')
    df_prom_categoria['Media_por_Replica'] = df_prom_categoria['Total_Resueltas'] / NUM_REPLICAS
    df_prom_categoria = df_prom_categoria.sort_values(by='Media_por_Replica', ascending=False)
    
    plt.figure(figsize=(12, 6))
    # 🟢 CORRECCIÓN DE WARNING: Añadido hue='Categoria' y legend=False
    sns.barplot(x='Categoria', y='Media_por_Replica', data=df_prom_categoria, hue='Categoria', palette='Spectral', legend=False)
    plt.title(f'Quejas Resueltas por Categoría (Media por Réplica)', fontsize=16) 
    plt.xlabel('Categoría de Queja')
    plt.ylabel('Media de Quejas Resueltas por Réplica')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    

else:
    print("\nNo se pudo consolidar la información de las réplicas para el análisis.")