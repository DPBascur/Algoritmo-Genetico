import numpy as np
from scipy.spatial import ConvexHull
import random

# Parámetros del problema
diputados = 431
quorum = 217
poblacion_size = 100
generaciones = 10000
prob_mutacion = 0.15

# Simulación de posiciones políticas en 2D (puedes cambiar la dimensión)
posiciones = np.random.rand(diputados, 2)
# Matriz de distancias euclidianas
from scipy.spatial.distance import pdist, squareform
distancias = squareform(pdist(posiciones, metric='euclidean'))

# Función de fitness: suma de distancias entre miembros de la coalición
def fitness(cromosoma):
    indices = np.where(cromosoma == 1)[0]
    if len(indices) < quorum:
        return 1e9  # Penalización fuerte si no cumple quórum
    submatriz = distancias[np.ix_(indices, indices)]
    return np.sum(submatriz) / 2  # Dividir entre 2 porque la matriz es simétrica

# Inicialización de la población
def inicializar_poblacion():
    poblacion = []
    for _ in range(poblacion_size):
        cromosoma = np.zeros(diputados, dtype=int)
        seleccionados = np.random.choice(diputados, quorum, replace=False)
        cromosoma[seleccionados] = 1
        # Posible agregar más 1s aleatoriamente
        extra = np.random.randint(0, diputados - quorum + 1)
        if extra > 0:
            adicionales = np.random.choice(np.where(cromosoma == 0)[0], extra, replace=False)
            cromosoma[adicionales] = 1
        poblacion.append(cromosoma)
    return np.array(poblacion)

# Selección por torneo
def seleccion(poblacion, fitnesses):
    i, j = random.sample(range(poblacion_size), 2)
    return poblacion[i] if fitnesses[i] < fitnesses[j] else poblacion[j]

# Cruza de un punto
def cruza(padre1, padre2):
    punto = random.randint(1, diputados - 2)
    hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
    return hijo1, hijo2

# Reparación para cumplir quórum
def reparar(cromosoma):
    total = np.sum(cromosoma)
    if total < quorum:
        ceros = np.where(cromosoma == 0)[0]
        agregar = np.random.choice(ceros, quorum - total, replace=False)
        cromosoma[agregar] = 1
    elif total > quorum:
        unos = np.where(cromosoma == 1)[0]
        quitar = np.random.choice(unos, total - quorum, replace=False)
        cromosoma[quitar] = 0
    return cromosoma

# Mutación
def mutar(cromosoma):
    # Vectoriza la mutación
    mask = np.random.rand(diputados) < prob_mutacion
    cromosoma = np.where(mask, 1 - cromosoma, cromosoma)
    return reparar(cromosoma)

# Algoritmo genético principal
def algoritmo_genetico():
    poblacion = inicializar_poblacion()
    mejor_fitness = 1e9
    mejor_cromosoma = None
    for gen in range(generaciones):
        fitnesses = np.array([fitness(c) for c in poblacion])
        idx_mejor = np.argmin(fitnesses)
        if fitnesses[idx_mejor] < mejor_fitness:
            mejor_fitness = fitnesses[idx_mejor]
            mejor_cromosoma = poblacion[idx_mejor].copy()
        nueva_poblacion = [mejor_cromosoma]  # Elitismo
        while len(nueva_poblacion) < poblacion_size:
            padre1 = seleccion(poblacion, fitnesses)
            padre2 = seleccion(poblacion, fitnesses)
            hijo1, hijo2 = cruza(padre1, padre2)
            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2)
            nueva_poblacion.extend([hijo1, hijo2])
        poblacion = np.array(nueva_poblacion[:poblacion_size])
        if True:  # Mostrar en cada generación
            tam_coalicion = np.sum(mejor_cromosoma)
            print(f"Generación {gen+1}: Mejor fitness = {mejor_fitness:.2f} | Tamaño coalición = {tam_coalicion}")
    return mejor_cromosoma, mejor_fitness

# Ejecutar el algoritmo
def main():
    mejor_cromosoma, mejor_fitness = algoritmo_genetico()
    print("\nMejor coalición encontrada:")
    miembros = np.where(mejor_cromosoma == 1)[0]
    print(f"Diputados en la coalición: {miembros.tolist()}")
    print(f"Tamaño de la coalición: {len(miembros)}")
    print(f"Fitness (suma de distancias): {mejor_fitness:.2f}")
    # Polígono convexo
    puntos_coalicion = posiciones[mejor_cromosoma == 1]
    if len(puntos_coalicion) > 2:
        hull = ConvexHull(puntos_coalicion)
        print("Vértices del polígono convexo:")
        print(puntos_coalicion[hull.vertices])
    else:
        print("No se puede calcular el polígono convexo (menos de 3 puntos)")

if __name__ == "__main__":
    main()
