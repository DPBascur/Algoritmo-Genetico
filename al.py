import json
import random
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

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

# Generar individuo aleatorio válido (exactamente quorum unos)
def generar_individuo():
    individuo = np.array([0] * total_diputados, dtype=int)
    indices = random.sample(range(total_diputados), quorum)
    individuo[indices] = 1
    return individuo

# Crear población inicial
def crear_poblacion(n):
    return [generar_individuo() for _ in range(n)]

# Mutación por swap: intercambia un 1 y un 0
def mutar_swap(individuo, tasa=0.17):
    if random.random() < tasa:
        ones = np.flatnonzero(individuo == 1)
        zeros = np.flatnonzero(individuo == 0)
        if len(ones) > 0 and len(zeros) > 0:
            i = random.choice(ones)
            j = random.choice(zeros)
            individuo[i], individuo[j] = 0, 1
    return individuo

# Cruce basado en conjuntos: hijo = muestra aleatoria de la unión de los unos de ambos padres
def cruce_conjuntos(p1, p2):
    set1 = set(np.flatnonzero(p1))
    set2 = set(np.flatnonzero(p2))
    union = list(set1 | set2)
    if len(union) < quorum:
        # Si la unión es menor que quorum, rellena con random
        union += random.sample(list(set(range(total_diputados)) - set(union)), quorum - len(union))
    hijo = np.array([0] * total_diputados, dtype=int)
    indices = random.sample(union, quorum)
    hijo[indices] = 1
    return hijo

# Selección probabilística eficiente
def seleccion_probabilistica(poblacion_ordenada, probs):
    idx = np.random.choice(len(poblacion_ordenada), p=probs)
    return poblacion_ordenada[idx]

# Algoritmo genético principal
def algoritmo_genetico(generaciones=10000, tamaño_poblacion=38):
    poblacion = crear_poblacion(tamaño_poblacion)
    mejor = min(poblacion, key=calcular_fitness)
    p = 0.141
    gen_encontrado = None
    for gen in range(generaciones):
        fitness_dict = {ind.tobytes(): calcular_fitness(ind) for ind in poblacion}
        poblacion_ordenada = sorted(poblacion, key=lambda ind: fitness_dict[ind.tobytes()])
        n = len(poblacion_ordenada)
        probs = np.array([p * (1 - p) ** i for i in range(n)])
        probs /= probs.sum()
        nueva_poblacion = [poblacion_ordenada[0]]  # Elitismo
        while len(nueva_poblacion) < tamaño_poblacion:
            padre1 = seleccion_probabilistica(poblacion_ordenada, probs)
            padre2 = seleccion_probabilistica(poblacion_ordenada, probs)
            hijo = cruce_conjuntos(padre1, padre2)
            hijo = mutar_swap(hijo, tasa=0.17)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion[:tamaño_poblacion]
        mejor_en_generacion = min(poblacion, key=calcular_fitness)
        if calcular_fitness(mejor_en_generacion) < calcular_fitness(mejor):
            mejor = mejor_en_generacion
        print(f"Generación {gen+1}, mejor fitness: {calcular_fitness(mejor):.5f}, tamaño coalición: {sum(mejor)}")
        if calcular_fitness(mejor) <= 9690:
            gen_encontrado = gen + 1
            print(f"\n¡Fitness objetivo 9690 o mejor alcanzado en la generación {gen_encontrado}!")
            break
    return mejor, gen_encontrado

# Ejecutar
mejor_individuo, gen_encontrado = algoritmo_genetico()

# Mostrar resultados
coalicion = [diputados_yea[i] for i, bit in enumerate(mejor_individuo) if bit == 1]
print("\nCoalición Ganadora Mínima (MWC):")
for dip in coalicion:
    print(f"{dip['name']} ({dip['party_short_name']}) - {dip['state_abbrev']}")
if gen_encontrado:
    print(f"\nFitness <= 9690 alcanzado en la generación: {gen_encontrado}")
else:
    print("No se alcanzó el fitness objetivo en las generaciones dadas.")

# Polígono convexo
def graficar_coalicion(coalicion, todos):
    coords_coal = np.array([[dip['x'], dip['y']] for dip in coalicion])
    coords_todos = np.array([[dip['x'], dip['y']] for dip in todos])
    hull = ConvexHull(coords_coal)
    plt.figure(figsize=(9, 7))
    # Todos los congresistas en gris
    plt.scatter(coords_todos[:, 0], coords_todos[:, 1], c='lightgray', label='Todos los congresistas', alpha=0.6)
    # Coalición en azul
    plt.scatter(coords_coal[:, 0], coords_coal[:, 1], c='blue', label='Coalición ganadora')
    # Polígono convexo
    for simplex in hull.simplices:
        plt.plot(coords_coal[simplex, 0], coords_coal[simplex, 1], 'r-')
    plt.plot(coords_coal[hull.vertices, 0], coords_coal[hull.vertices, 1], 'r--', lw=2, label='Polígono convexo')
    plt.scatter(coords_coal[hull.vertices, 0], coords_coal[hull.vertices, 1], c='red', s=60, label='Vértices')
    plt.title('Coalición Ganadora Mínima y su Polígono Convexo')
    plt.xlabel('DW-Nominate Dimension 1: Economic/Redistributive')
    plt.ylabel('NOMINATE Dimension 2: Other Votes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

graficar_coalicion(coalicion, diputados_yea)