import numpy as np
from itertools import combinations
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ==========================================
# PARÁMETROS DEL ALGORITMO GENÉTICO
# ==========================================

s_Datos = "VoteData.json"
i_tamanoPoblacion = 38
i_tamanoCoalicion = 217
i_generaciones = 10000
f_tasaMutacion = 0.1700019
f_porcentajeDiversidad = 0.2  # 20% de reinyección cada 10 generaciones

# ==========================================
# FUNCIONES
# ==========================================

def cargarDatosJson(s_Datos: str) -> pd.DataFrame:
    with open(s_Datos, "r", encoding="utf-8") as f_archivo:
        d_json = json.load(f_archivo)
    l_votacion = d_json["rollcalls"][0]["votes"]
    l_registros = []
    for d in l_votacion:
        l_registros.append({
            "i_id": d.get("icpsr"),
            "s_nombre": d.get("name"),
            "s_estado": d.get("state_abbrev"),
            "s_partido": d.get("party"),
            "s_voto": d.get("vote"),
            "f_x": d.get("x"),
            "f_y": d.get("y")
        })
    df_datos = pd.DataFrame(l_registros)
    return df_datos.dropna(subset=["f_x", "f_y"])


def f_calcularFitnessCoalicion(l_iIDsCoalicion: list, df_d_diputados: pd.DataFrame) -> float:
    df_d_coalicion = df_d_diputados[df_d_diputados["i_id"].isin(l_iIDsCoalicion)]
    a_f_puntos = df_d_coalicion[["f_x", "f_y"]].values
    return sum(np.linalg.norm(a_f_p1 - a_f_p2)
               for a_f_p1, a_f_p2 in combinations(a_f_puntos, 2))


def f_generarPoblacionInicial(i_tamanoPoblacion: int, i_tamanoCoalicion: int, l_i_idsDisponibles: list) -> list:
    return [random.sample(l_i_idsDisponibles, i_tamanoCoalicion) for _ in range(i_tamanoPoblacion)]


def f_seleccionRuletaInversa(l_l_poblacion: list, l_f_fitness: list, i_numeroSeleccionados: int) -> list:
    f_epsilon = 1e-6
    a_f_inversos = np.array([1 / (f + f_epsilon) for f in l_f_fitness])
    a_f_probabilidades = a_f_inversos / np.sum(a_f_inversos)
    l_indicesSeleccionados = np.random.choice(len(l_l_poblacion), size=i_numeroSeleccionados, p=a_f_probabilidades, replace=True)
    return [l_l_poblacion[i] for i in l_indicesSeleccionados]


def f_cruzarPadres(l_i_padre1: list, l_i_padre2: list, i_tamanoCoalicion: int, l_i_idsValidos: list) -> list:
    s_padre1, s_padre2 = set(l_i_padre1), set(l_i_padre2)
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
        l_i_mutada = l_i_coalicion.copy()
        i_fuera = random.choice([i for i in l_i_idsValidos if i not in l_i_mutada])
        i_dentro = random.choice(l_i_mutada)
        l_i_mutada.remove(i_dentro)
        l_i_mutada.append(i_fuera)
        return l_i_mutada
    return l_i_coalicion

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

df_diputados = cargarDatosJson(s_Datos)
df_activos = df_diputados[df_diputados["s_voto"] == "Yea"]
l_i_idsValidos = df_activos["i_id"].tolist()

l_l_poblacion = f_generarPoblacionInicial(i_tamanoPoblacion, i_tamanoCoalicion, l_i_idsValidos)
f_mejorGlobalFitness = float("inf")
l_i_mejorGlobalCoalicion = []

for i_generacion in range(i_generaciones):
    l_f_fitness = [f_calcularFitnessCoalicion(ind, df_activos) for ind in l_l_poblacion]
    f_mejorFitness = min(l_f_fitness)
    i_mejorIndex = l_f_fitness.index(f_mejorFitness)

    if f_mejorFitness < f_mejorGlobalFitness:
        f_mejorGlobalFitness = f_mejorFitness
        l_i_mejorGlobalCoalicion = l_l_poblacion[i_mejorIndex].copy()

    print(f"[Generación {i_generacion+1}] Mejor fitness: {f_mejorFitness:.2f}")

    if i_generacion > 0 and i_generacion % 10 == 0:
        i_nuevos = int(i_tamanoPoblacion * f_porcentajeDiversidad)
        l_l_poblacion[:i_nuevos] = f_generarPoblacionInicial(i_nuevos, i_tamanoCoalicion, l_i_idsValidos)

    l_l_padres = f_seleccionRuletaInversa(l_l_poblacion, l_f_fitness, i_tamanoPoblacion)
    l_l_hijos = [f_cruzarPadres(random.choice(l_l_padres), random.choice(l_l_padres),
                                i_tamanoCoalicion, l_i_idsValidos) for _ in range(i_tamanoPoblacion)]
    l_l_mutados = [f_mutarIndividuo(hijo, l_i_idsValidos, f_tasaMutacion) for hijo in l_l_hijos]
    i_pos = random.randint(0, i_tamanoPoblacion - 1)
    l_l_mutados[i_pos] = l_i_mejorGlobalCoalicion.copy()
    l_l_poblacion = l_l_mutados

# ==========================================
# RESULTADOS FINALES
# ==========================================

print(f"\n🎉 Mejor fitness global: {f_mejorGlobalFitness:.2f}")
df_mwc = df_activos[df_activos["i_id"].isin(l_i_mejorGlobalCoalicion)].copy()
a_f_puntos = df_mwc[["f_x", "f_y"]].values
o_hull = ConvexHull(a_f_puntos)
df_vertices = df_mwc.iloc[o_hull.vertices].copy()

# Guardar como CSV
df_mwc.to_csv("miembros_MWC.csv", index=False)
df_vertices.to_csv("vertices_poligono_convexo.csv", index=False)

# Visualización
plt.figure(figsize=(8, 6))
plt.scatter(df_mwc["f_x"], df_mwc["f_y"], alpha=0.6, label="Miembros MWC")
hull_pts = df_vertices[["f_x", "f_y"]].values
hull_pts = np.append(hull_pts, [hull_pts[0]], axis=0)
plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'r-', linewidth=2, label="Polígono Convexo")
plt.title("Coalición Ganadora Mínima y su Polígono Convexo")
plt.xlabel("DW-NOMINATE X")
plt.ylabel("DW-NOMINATE Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
