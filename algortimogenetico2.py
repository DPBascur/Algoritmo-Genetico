import numpy as np
import json
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import random

# Parámetros
RUTA_JSON = "VoteData.json"
TAMANO_POBLACION = 38
TAMANO_COALICION = 217
GENERACIONES = 10000
TASA_MUTACION = 0.17
PORCENTAJE_DIVERSIDAD = 0.2

# 1. Cargar datos y filtrar diputados activos
with open(RUTA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

votos = data["rollcalls"][0]["votes"]
diputados = [d for d in votos if d["vote"] == "Yea" and d["x"] is not None and d["y"] is not None]

ids = np.array([d["icpsr"] for d in diputados])
coords = np.array([[d["x"], d["y"]] for d in diputados])
n_diputados = len(ids)

# 2. Función de fitness vectorizada

def fitness(individuo):
    puntos = coords[individuo]
    if len(puntos) < 2:
        return np.inf
    return pdist(puntos).sum()

# 3. Inicialización de la población

def generar_poblacion():
    poblacion = []
    while len(poblacion) < TAMANO_POBLACION:
        seleccion = np.zeros(n_diputados, dtype=bool)
        seleccion[random.sample(range(n_diputados), TAMANO_COALICION)] = True
        poblacion.append(seleccion)
    return np.array(poblacion)

# 4. Selección por ruleta inversa

def seleccion_ruleta(poblacion, fitnesses):
    inv_fit = 1 / (fitnesses + 1e-6)
    probs = inv_fit / inv_fit.sum()
    idxs = np.random.choice(len(poblacion), size=len(poblacion), p=probs)
    return poblacion[idxs]

# 5. Cruzamiento (intercambio de genes)

def cruzar(padre1, padre2):
    hijo = padre1.copy()
    mask = np.random.rand(n_diputados) < 0.5
    hijo[mask] = padre2[mask]
    # Ajustar tamaño
    while hijo.sum() > TAMANO_COALICION:
        idx = np.where(hijo)[0]
        hijo[random.choice(idx)] = False
    while hijo.sum() < TAMANO_COALICION:
        idx = np.where(~hijo)[0]
        hijo[random.choice(idx)] = True
    return hijo

# 6. Mutación

def mutar(individuo):
    if random.random() < TASA_MUTACION:
        idx_in = np.where(individuo)[0]
        idx_out = np.where(~individuo)[0]
        if len(idx_in) > 0 and len(idx_out) > 0:
            individuo[random.choice(idx_in)] = False
            individuo[random.choice(idx_out)] = True
    return individuo

# 7. Algoritmo genético principal

def algoritmo_genetico():
    poblacion = generar_poblacion()
    mejor_fitness = np.inf
    mejor_individuo = None
    for gen in range(GENERACIONES):
        fitnesses = np.array([fitness(ind) for ind in poblacion])
        idx_best = np.argmin(fitnesses)
        if fitnesses[idx_best] < mejor_fitness:
            mejor_fitness = fitnesses[idx_best]
            mejor_individuo = poblacion[idx_best].copy()
        print(f"[Gen {gen+1}] Mejor fitness: {fitnesses[idx_best]:.2f}")
        # Diversidad
        if gen > 0 and gen % 10 == 0:
            n_nuevos = int(TAMANO_POBLACION * PORCENTAJE_DIVERSIDAD)
            nuevos = generar_poblacion()[:n_nuevos]
            poblacion[:n_nuevos] = nuevos
        padres = seleccion_ruleta(poblacion, fitnesses)
        hijos = []
        for _ in range(TAMANO_POBLACION):
            p1, p2 = padres[random.randint(0, TAMANO_POBLACION-1)], padres[random.randint(0, TAMANO_POBLACION-1)]
            hijo = cruzar(p1, p2)
            hijo = mutar(hijo)
            hijos.append(hijo)
        # Elitismo
        hijos[random.randint(0, TAMANO_POBLACION-1)] = mejor_individuo.copy()
        poblacion = np.array(hijos)
    return mejor_individuo, mejor_fitness

# 8. Ejecutar y mostrar resultados
if __name__ == "__main__":
    mejor, fit = algoritmo_genetico()
    print("\nMejor fitness global:", fit)
    miembros = ids[mejor]
    print("Miembros de la MWC:", miembros)
    puntos = coords[mejor]
    hull = ConvexHull(puntos)
    print("Vértices del polígono convexo:", miembros[hull.vertices])
