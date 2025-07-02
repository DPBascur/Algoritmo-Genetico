import json
import random
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

rng = np.random.default_rng()

# Cargar datos desde JSON
with open("VoteData.json", "r", encoding="utf-8") as f:
    data = json.load(f)

diputados_yea = [v for v in data['rollcalls'][0]['votes']]
total_diputados = len(diputados_yea)
total_votantes = len(data['rollcalls'][0]['votes'])
quorum = total_votantes // 2 + 1

# Función de fitness optimizada con pdist
def calcular_fitness(individuo):
    seleccionados = np.array(diputados_yea)[np.flatnonzero(individuo)]
    if len(seleccionados) < quorum:
        return float('inf')
    coords = np.array([[d['x'], d['y']] for d in seleccionados])
    return pdist(coords).sum()

# Reparar cromosoma si no cumple el quorum
def corregir(individuo):
    suma = sum(individuo)
    dif = abs(suma - quorum)

    if suma < quorum:
        indices = np.flatnonzero(individuo == 0)
        sel = rng.choice(indices, size=dif, replace=False)
        individuo[sel] = 1
    else:
        indices = np.flatnonzero(individuo == 1)
        sel = rng.choice(indices, size=dif, replace=False)
        individuo[sel] = 0

    return individuo

# Generar individuo aleatorio válido
def generar_individuo():
    individuo = np.array([0] * total_diputados, dtype=int)
    indices = random.sample(range(total_diputados), quorum + random.randint(0, 10))
    individuo[indices] = 1
    return individuo

# Crear población inicial
def crear_poblacion(n):
    return [generar_individuo() for _ in range(n)]

# Mutación simple con tasa reducida
def mutar(individuo, tasa=0.15):
    mutacion = (np.random.random(total_diputados) < tasa).astype(int)
    return individuo ^ mutacion

# Cruce y reparación
def cruce(p1, p2):
    punto = random.randint(1, total_diputados - 2)
    hijo1 = np.concatenate((p1[:punto], p2[punto:]))
    hijo2 = np.concatenate((p2[:punto], p1[punto:]))

    hijo1 = corregir(mutar(hijo1))
    hijo2 = corregir(mutar(hijo2))
    return hijo1, hijo2

# Selección probabilística eficiente
def seleccion_probabilistica(poblacion_ordenada, probs):
    idx = np.random.choice(len(poblacion_ordenada), p=probs)
    return poblacion_ordenada[idx]

# Algoritmo genético principal
def algoritmo_genetico(generaciones=50000, tamaño_poblacion=50):
    poblacion = crear_poblacion(tamaño_poblacion)
    mejor = min(poblacion, key=calcular_fitness)

    for gen in range(generaciones):
        # Calcular fitness y ordenar
        fitness_dict = {ind.tobytes(): calcular_fitness(ind) for ind in poblacion}
        poblacion_ordenada = sorted(poblacion, key=lambda ind: fitness_dict[ind.tobytes()])

        # Probabilidades exponenciales (solo una vez)
        p = 0.141
        n = len(poblacion_ordenada)
        probs = np.array([p * (1 - p) ** i for i in range(n)])
        probs /= probs.sum()

        nueva_poblacion = [poblacion_ordenada[0]]  # Elitismo

        while len(nueva_poblacion) < tamaño_poblacion:
            padre1 = seleccion_probabilistica(poblacion_ordenada, probs)
            padre2 = seleccion_probabilistica(poblacion_ordenada, probs)
            hijo1, hijo2 = cruce(padre1, padre2)

            if len(nueva_poblacion) < tamaño_poblacion:
                nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tamaño_poblacion:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

        mejor_en_generacion = min(poblacion, key=calcular_fitness)
        if calcular_fitness(mejor_en_generacion) < calcular_fitness(mejor):
            mejor = mejor_en_generacion

        print(f"Generación {gen+1}, mejor fitness: {calcular_fitness(mejor):.2f}, tamaño coalición: {sum(mejor)}")

    return mejor

# Ejecutar
mejor_individuo = algoritmo_genetico()

# Mostrar resultados
coalicion = [diputados_yea[i] for i, bit in enumerate(mejor_individuo) if bit == 1]
print("\nCoalición Ganadora Mínima (MWC):")
for dip in coalicion:
    print(f"{dip['name']} ({dip['party_short_name']}) - {dip['state_abbrev']}")

# Polígono convexo
coords = np.array([[dip['x'], dip['y']] for dip in coalicion])
hull = ConvexHull(coords)
print("\nVértices del polígono convexo:")
for i in hull.vertices:
    dip = coalicion[i]
    print(f"{dip['name']} - ({dip['x']:.3f}, {dip['y']:.3f})")