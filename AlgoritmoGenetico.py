import numpy as np
from itertools import combinations
import json
import pandas as pd
import random

# Ruta del archivo JSON con los datos de votación
s_Datos = "VoteData.json"

# ==========================================
# FUNCIONES
# ==========================================

def cargarDatosJson(s_Datos: str) -> pd.DataFrame:
    with open(s_Datos, "r", encoding="utf-8") as f_archivo:
        d_json = json.load(f_archivo)

    l_votacion = d_json["rollcalls"][0]["votes"]
    l_registrosEstructurados = []

    for d_diputado in l_votacion:
        d_registro = {
            "i_id": d_diputado.get("icpsr"),
            "s_nombre": d_diputado.get("name"),
            "s_estado": d_diputado.get("state_abbrev"),
            "s_partido": d_diputado.get("party"),
            "s_voto": d_diputado.get("vote"),
            "f_x": d_diputado.get("x"),
            "f_y": d_diputado.get("y")
        }
        l_registrosEstructurados.append(d_registro)

    df_datos = pd.DataFrame(l_registrosEstructurados)
    df_datos = df_datos.dropna(subset=["f_x", "f_y"])
    return df_datos


def f_calcularFitnessCoalicion(l_iIDsCoalicion: list, df_d_diputados: pd.DataFrame) -> float:
    df_d_coalicion = df_d_diputados[df_d_diputados["i_id"].isin(l_iIDsCoalicion)]
    a_f_puntos = df_d_coalicion[["f_x", "f_y"]].values
    f_valorFitness = 0.0
    for a_f_p1, a_f_p2 in combinations(a_f_puntos, 2):
        f_distancia = np.linalg.norm(a_f_p1 - a_f_p2)
        f_valorFitness += f_distancia
    return f_valorFitness


def f_generarPoblacionInicial(i_tamanoPoblacion: int, i_tamanoCoalicion: int, l_i_idsDisponibles: list) -> list:
    l_l_poblacion = []
    for _ in range(i_tamanoPoblacion):
        l_i_coalicion = random.sample(l_i_idsDisponibles, i_tamanoCoalicion)
        l_l_poblacion.append(l_i_coalicion)
    return l_l_poblacion


def f_seleccionRuletaInversa(l_l_poblacion: list, l_f_fitness: list, i_numeroSeleccionados: int) -> list:
    f_epsilon = 1e-6
    a_f_inversos = np.array([1 / (f + f_epsilon) for f in l_f_fitness])
    a_f_probabilidades = a_f_inversos / np.sum(a_f_inversos)
    l_indicesSeleccionados = np.random.choice(len(l_l_poblacion), size=i_numeroSeleccionados, p=a_f_probabilidades, replace=True)
    l_l_seleccionados = [l_l_poblacion[i] for i in l_indicesSeleccionados]
    return l_l_seleccionados


def f_cruzarPadres(l_i_padre1: list, l_i_padre2: list, i_tamanoCoalicion: int, l_i_idsValidos: list) -> list:
    s_padre1 = set(l_i_padre1)
    s_padre2 = set(l_i_padre2)
    s_comunes = s_padre1.intersection(s_padre2)
    s_union_restante = list((s_padre1.union(s_padre2)) - s_comunes)
    random.shuffle(s_union_restante)
    l_i_hijo = list(s_comunes)
    while len(l_i_hijo) < i_tamanoCoalicion and s_union_restante:
        l_i_hijo.append(s_union_restante.pop())
    while len(l_i_hijo) < i_tamanoCoalicion:
        i_candidato = random.choice(l_i_idsValidos)
        if i_candidato not in l_i_hijo:
            l_i_hijo.append(i_candidato)
    return l_i_hijo


def f_mutarIndividuo(l_i_coalicion: list, l_i_idsValidos: list, f_tasaMutacion: float) -> list:
    if random.random() < f_tasaMutacion:
        l_i_coalicionMutada = l_i_coalicion.copy()
        i_fuera = random.choice([i for i in l_i_idsValidos if i not in l_i_coalicionMutada])
        i_dentro = random.choice(l_i_coalicionMutada)
        l_i_coalicionMutada.remove(i_dentro)
        l_i_coalicionMutada.append(i_fuera)
        return l_i_coalicionMutada
    else:
        return l_i_coalicion

# ==========================================
# PARÁMETROS DEL ALGORITMO GENÉTICO
# ==========================================

i_tamanoPoblacion = 38
i_tamanoCoalicion = 217
i_generaciones = 1000
f_tasaMutacion = 0.1700019 #0.15  AUMENTADA para evitar estancamiento
f_porcentajeDiversidad = 0.2  # 20% de reinyección cada 10 generaciones

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

df_diputadosProcesado = cargarDatosJson(s_Datos)
df_d_activos = df_diputadosProcesado[df_diputadosProcesado["s_voto"] == "Yea"]
l_i_idsValidos = df_d_activos["i_id"].tolist()

l_l_poblacion = f_generarPoblacionInicial(i_tamanoPoblacion, i_tamanoCoalicion, l_i_idsValidos)

f_mejorGlobalFitness = float("inf")
l_i_mejorGlobalCoalicion = []

for i_generacion in range(i_generaciones):
    l_f_fitness = [f_calcularFitnessCoalicion(l_ind, df_d_activos) for l_ind in l_l_poblacion]
    f_mejorFitness = min(l_f_fitness)
    i_mejorIndex = l_f_fitness.index(f_mejorFitness)

    if f_mejorFitness < f_mejorGlobalFitness:
        f_mejorGlobalFitness = f_mejorFitness
        l_i_mejorGlobalCoalicion = l_l_poblacion[i_mejorIndex].copy()

    print(f"[Generación {i_generacion+1}] Mejor fitness: {f_mejorFitness:.2f}")

    # 🔁 Reinyección de diversidad cada 10 generaciones
    if i_generacion > 0 and i_generacion % 10 == 0:
        i_nuevos = int(i_tamanoPoblacion * f_porcentajeDiversidad)
        l_l_reemplazo = f_generarPoblacionInicial(i_nuevos, i_tamanoCoalicion, l_i_idsValidos)
        l_l_poblacion[:i_nuevos] = l_l_reemplazo

    l_l_padres = f_seleccionRuletaInversa(l_l_poblacion, l_f_fitness, i_tamanoPoblacion)

    l_l_hijos = []
    for _ in range(i_tamanoPoblacion):
        l_i_padre1 = random.choice(l_l_padres)
        l_i_padre2 = random.choice(l_l_padres)
        l_i_hijo = f_cruzarPadres(l_i_padre1, l_i_padre2, i_tamanoCoalicion, l_i_idsValidos)
        l_l_hijos.append(l_i_hijo)

    l_l_mutados = [f_mutarIndividuo(l_hijo, l_i_idsValidos, f_tasaMutacion) for l_hijo in l_l_hijos]

    i_pos = random.randint(0, i_tamanoPoblacion - 1)
    l_l_mutados[i_pos] = l_i_mejorGlobalCoalicion.copy()

    l_l_poblacion = l_l_mutados

# Resultado final
print("\n🎉 Mejor fitness global:", f_mejorGlobalFitness)
