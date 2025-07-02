import numpy as np
import json
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd

# ================================
# PARÁMETROS DEL ALGORITMO
# ================================
s_Datos = "VoteData.json"
i_tamanoPoblacion = 38
i_tamanoCoalicion = 217
i_generaciones = 10000
f_tasaMutacion = 0.15

# ================================
# FUNCIONES AUXILIARES
# ================================
def cargarDatosComoVectores(s_rutaArchivo: str):
    with open(s_rutaArchivo, "r", encoding="utf-8") as f:
        d_json = json.load(f)

    l_votacion = d_json["rollcalls"][0]["votes"]
    a_ids, a_nombres, a_coords = [], [], []
    for d in l_votacion:
        if d["vote"] == "Yea" and d["x"] is not None and d["y"] is not None:
            a_ids.append(d["icpsr"])
            a_nombres.append(d["name"])
            a_coords.append([d["x"], d["y"]])
    return np.array(a_ids), np.array(a_nombres), np.array(a_coords)

def f_fitness(a_i_indices, a_f_coords):
    a_f_subcoords = a_f_coords[a_i_indices]
    diff = a_f_subcoords[:, np.newaxis, :] - a_f_subcoords[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return np.sum(np.triu(dist, k=1))  # Evita duplicados y diagonales

def f_generarPoblacion(i_n, i_tam, i_total):
    return [np.random.choice(i_total, size=i_tam, replace=False) for _ in range(i_n)]

def f_ruletaInversa(a_f_fitness):
    inv = 1.0 / (a_f_fitness + 1e-6)
    probas = inv / np.sum(inv)
    return np.random.choice(len(a_f_fitness), size=len(a_f_fitness), p=probas)

def f_cruzar(ind1, ind2, i_tamano, i_max):
    s1, s2 = set(ind1), set(ind2)
    comunes = list(s1 & s2)
    restantes = list((s1 | s2) - set(comunes))
    random.shuffle(restantes)
    hijo = comunes + restantes
    while len(hijo) < i_tamano:
        c = random.randint(0, i_max - 1)
        if c not in hijo:
            hijo.append(c)
    return np.array(hijo[:i_tamano])

def f_mutar(individuo, i_max, f_tasa):
    if random.random() < f_tasa:
        nuevo = individuo.copy()
        disponibles = list(set(range(i_max)) - set(nuevo))
        if disponibles:
            fuera = random.choice(disponibles)
            dentro_idx = random.randint(0, len(nuevo) - 1)
            nuevo[dentro_idx] = fuera
        return nuevo
    return individuo

# ================================
# EJECUCIÓN PRINCIPAL
# ================================
a_i_ids, a_s_nombres, a_f_coords = cargarDatosComoVectores(s_Datos)
i_totalDiputados = len(a_i_ids)

l_l_poblacion = f_generarPoblacion(i_tamanoPoblacion, i_tamanoCoalicion, i_totalDiputados)
f_mejorFitnessGlobal = float("inf")
a_i_mejorMWC = None

# Configurar visualización interactiva
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

for g in range(i_generaciones):
    a_f_fitness = np.array([f_fitness(ind, a_f_coords) for ind in l_l_poblacion])
    i_mejorGen = np.argmin(a_f_fitness)
    if a_f_fitness[i_mejorGen] < f_mejorFitnessGlobal:
        f_mejorFitnessGlobal = a_f_fitness[i_mejorGen]
        a_i_mejorMWC = l_l_poblacion[i_mejorGen].copy()
    
    # Visualización dinámica cada 100 generaciones
    if g % 100 == 0 or g == i_generaciones - 1:
        print(f"[Generación {g+1}] Mejor fitness: {a_f_fitness[i_mejorGen]:.2f}")
        
        ax.clear()
        
        # Graficar todos los individuos
        for ind in l_l_poblacion:
            coords = a_f_coords[ind]
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.05, color='gray', s=10)
        
        # Graficar el mejor hasta ahora
        best_coords = a_f_coords[a_i_mejorMWC]
        ax.scatter(best_coords[:, 0], best_coords[:, 1], color='red', s=20, label="Mejor MWC")

        ax.set_title(f"Generación {g+1} - Evolución MWC")
        ax.set_xlabel("DW-NOMINATE X")
        ax.set_ylabel("DW-NOMINATE Y")
        ax.legend()
        ax.grid(True)
        plt.pause(0.01)

    i_padres = f_ruletaInversa(a_f_fitness)
    padres_seleccionados = [l_l_poblacion[i] for i in i_padres]

    hijos = [f_cruzar(random.choice(padres_seleccionados),
                     random.choice(padres_seleccionados),
                     i_tamanoCoalicion, i_totalDiputados)
             for _ in range(i_tamanoPoblacion)]

    mutados = [f_mutar(hijo, i_totalDiputados, f_tasaMutacion) for hijo in hijos]
    i_elite = random.randint(0, i_tamanoPoblacion - 1)
    mutados[i_elite] = a_i_mejorMWC.copy()
    l_l_poblacion = mutados

# Finalizar modo interactivo
plt.ioff()

# ================================
# EXPORTAR RESULTADOS Y VISUALIZAR
# ================================
print(f"\n🎯 Mejor fitness global: {f_mejorFitnessGlobal:.2f}")
a_f_mwc_coords = a_f_coords[a_i_mejorMWC]
a_s_mwc_nombres = a_s_nombres[a_i_mejorMWC]

df_mwc = pd.DataFrame({
    "i_id": a_i_ids[a_i_mejorMWC],
    "s_nombre": a_s_mwc_nombres,
    "f_x": a_f_mwc_coords[:, 0],
    "f_y": a_f_mwc_coords[:, 1]
})
df_mwc.to_csv("miembros_MWC_vectores.csv", index=False)

# Polígono convexo
o_hull = ConvexHull(a_f_mwc_coords)
df_vertices = df_mwc.iloc[o_hull.vertices]
df_vertices.to_csv("vertices_poligono_convexo_vectores.csv", index=False)

# Visualización final del mejor conjunto
plt.figure(figsize=(8, 6))
plt.scatter(a_f_mwc_coords[:, 0], a_f_mwc_coords[:, 1], alpha=0.6, label="Miembros MWC")
hull_pts = a_f_mwc_coords[o_hull.vertices]
hull_pts = np.append(hull_pts, [hull_pts[0]], axis=0)
plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'r-', linewidth=2, label="Polígono Convexo")
plt.title("Coalición Ganadora Mínima (Vectores) - Resultado Final")
plt.xlabel("DW-NOMINATE X")
plt.ylabel("DW-NOMINATE Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

