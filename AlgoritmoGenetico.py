import numpy as np  # Biblioteca para operaciones numéricas con arreglos
from scipy.spatial import ConvexHull  # Para calcular el polígono convexo
from scipy.spatial.distance import pdist, squareform  # Para calcular distancias euclidianas entre pares
import random  # Para generación de aleatoriedad
import json  # Para manejo de archivos JSON
import os  # Para manejo de rutas del sistema
import matplotlib.pyplot as plt  # Para generar gráficos
import csv  # Para guardar resultados en archivos CSV

# Parámetros globales del algoritmo genético

iDiputados = 431  # Número total de diputados
iQuorum = 216  # Cantidad mínima para formar una coalición válida (mayoría absoluta)
iPoblacionSize = 38  # Tamaño de la población en cada generación
iGeneraciones = 10000  # Número máximo de generaciones del algoritmo
tfProbMutacion = 0.1700019  # Probabilidad de aplicar mutación tipo swap en un cromosoma

# Función para cargar los datos desde archivo JSON
def fnCargarDatosJSON(strPath="VoteData.json"):
    try:
        fArchivo = open(strPath, 'r', encoding='utf-8')  # Intenta abrir archivo directamente
    except FileNotFoundError:
        strBase = os.path.dirname(__file__)  # Si falla, busca en el directorio del script actual
        fArchivo = open(os.path.join(strBase, strPath), 'r', encoding='utf-8')
    dictData = json.load(fArchivo)  # Cargar el contenido JSON como diccionario
    fArchivo.close()
    lstDiputados = [v for v in dictData['rollcalls'][0]['votes']]  # Extraer lista de diputados
    iTotalDiputados = len(lstDiputados)  # Contar diputados
    iQuorumCalculado = iTotalDiputados // 2 + 1  # Calcular quórum (mayoría absoluta)
    matPosiciones = np.array([[d['x'], d['y']] for d in lstDiputados])  # Posiciones (x, y) de cada diputado
    return lstDiputados, matPosiciones, iTotalDiputados, iQuorumCalculado

# Cargar datos iniciales
dlstDiputados, dmatPosiciones, diDiputados, diQuorum = fnCargarDatosJSON()
dmatDistancias = squareform(pdist(dmatPosiciones, metric='euclidean'))  # Matriz de distancias entre diputados

# Función de evaluación (fitness) de un cromosoma
def fnFitness(arrCromosoma):
    arrIdxMiembros = np.where(arrCromosoma == 1)[0]  # Obtener índices de diputados incluidos en la coalición
    if len(arrIdxMiembros) < diQuorum:
        return 1e9  # Penalización si no alcanza el quórum
    if len(arrIdxMiembros) < 2:
        return 0.0  # No hay distancia entre un solo individuo
    matDistSub = dmatDistancias[np.ix_(arrIdxMiembros, arrIdxMiembros)]  # Submatriz de distancias internas
    fSumaDistancias = matDistSub.sum() / 2.0  # Sumar distancias (dividir por 2 porque están duplicadas)
    return fSumaDistancias

# Inicialización de la población de cromosomas
def fnInicializarPoblacion():
    lstPoblacion = []
    for _ in range(iPoblacionSize):
        arrCromosoma = np.zeros(diDiputados, dtype=int)  # Crear vector binario de 0's
        arrSeleccionados = np.random.choice(diDiputados, diQuorum, replace=False)  # Selección aleatoria
        arrCromosoma[arrSeleccionados] = 1  # Marcar diputados seleccionados
        lstPoblacion.append(arrCromosoma)
    return np.array(lstPoblacion)

# Selección por torneo binario
def fnSeleccion(arrPoblacion, arrFitness):
    iIndex1, iIndex2 = random.sample(range(iPoblacionSize), 2)  # Selección aleatoria de 2 individuos
    return arrPoblacion[iIndex1] if arrFitness[iIndex1] < arrFitness[iIndex2] else arrPoblacion[iIndex2]

# Cruza que preserva intersección de padres y completa con genes únicos
def fnCruzaMejorado(arrPadre1, arrPadre2):
    arrComun = np.logical_and(arrPadre1, arrPadre2)  # Genes comunes entre padres
    iComun = np.sum(arrComun)
    arrHijo1 = arrComun.astype(int)
    arrHijo2 = arrComun.astype(int)
    iFaltantes = diQuorum - iComun
    if iFaltantes > 0:
        arrDiferencia = np.logical_xor(arrPadre1, arrPadre2)  # Genes únicos
        idxOpciones = np.where(arrDiferencia == 1)[0]
        np.random.shuffle(idxOpciones)  # Mezcla aleatoriamente los índices de opciones disponibles
        seleccionHijo1 = idxOpciones[:iFaltantes]  # Selecciona los primeros faltantes para el hijo 1
        seleccionHijo2 = idxOpciones[iFaltantes:2*iFaltantes] if len(idxOpciones) >= 2*iFaltantes else np.random.choice(idxOpciones, iFaltantes, replace=False)  # Selecciona para el hijo 2
        arrHijo1[seleccionHijo1] = 1  # Asigna los seleccionados al hijo 1
        arrHijo2[seleccionHijo2] = 1  # Asigna los seleccionados al hijo 2
    return arrHijo1, arrHijo2  # Devuelve ambos hijos

# Mutación por intercambio con probabilidad
def fnMutarSwapProb(arrCromosoma):
    if np.random.rand() < tfProbMutacion:  # Solo muta si se cumple la probabilidad
        arrMiembros = np.where(arrCromosoma == 1)[0]  # Índices de miembros actuales
        arrNoMiembros = np.where(arrCromosoma == 0)[0]  # Índices fuera de la coalición
        if len(arrMiembros) > 0 and len(arrNoMiembros) > 0:  # Si hay miembros y no-miembros
            iSalir = random.choice(arrMiembros)  # Elige un miembro al azar para salir
            iEntrar = random.choice(arrNoMiembros)  # Elige un no-miembro al azar para entrar
            arrCromosoma[iSalir] = 0  # Elimina el miembro seleccionado
            arrCromosoma[iEntrar] = 1  # Agrega el nuevo miembro
    return arrCromosoma  # Devuelve el cromosoma (mutado o no

# Ciclo principal del algoritmo genético
def fnAlgoritmoGenetico():
    arrPoblacion = fnInicializarPoblacion()  # Inicializa la población
    fMejorFitness = 1e9  # Inicializa el mejor fitness con un valor alto
    arrMejorCromosoma = None  # Inicializa el mejor cromosoma
    for iGen in range(iGeneraciones):  # Itera por el número de generaciones
        arrFitnessPob = np.array([fnFitness(crom) for crom in arrPoblacion])  # Calcula fitness de cada individuo
        iIdxMejor = np.argmin(arrFitnessPob)  # Índice del mejor individuo
        if arrFitnessPob[iIdxMejor] < fMejorFitness:  # Si el mejor de la generación es mejor que el global
            fMejorFitness = arrFitnessPob[iIdxMejor]  # Actualiza el mejor fitness global
            arrMejorCromosoma = arrPoblacion[iIdxMejor].copy()  # Actualiza el mejor cromosoma global
        lstNuevaPoblacion = [arrMejorCromosoma]  # Aplica elitismo: el mejor pasa directo
        while len(lstNuevaPoblacion) < iPoblacionSize:  # Completa la nueva población
            arrPadre1 = fnSeleccion(arrPoblacion, arrFitnessPob)  # Selecciona primer padre
            arrPadre2 = fnSeleccion(arrPoblacion, arrFitnessPob)  # Selecciona segundo padre
            arrHijo1, arrHijo2 = fnCruzaMejorado(arrPadre1, arrPadre2)  # Cruza padres para obtener hijos
            arrHijo1 = fnMutarSwapProb(arrHijo1)  # Aplica mutación al hijo 1
            arrHijo2 = fnMutarSwapProb(arrHijo2)  # Aplica mutación al hijo 2
            lstNuevaPoblacion.extend([arrHijo1, arrHijo2])  # Añade hijos a la nueva población
        arrPoblacion = np.array(lstNuevaPoblacion[:iPoblacionSize])  # Trunca si hay exceso de individuos
        print(f"Generación {iGen+1}: Mejor fitness = {fMejorFitness:.6f} | Tamaño coalición = {np.sum(arrMejorCromosoma)}")  # Muestra progreso
    return arrMejorCromosoma, fMejorFitness  # Devuelve el mejor cromosoma y su fitness

# Función principal de ejecución
def fnMain():
    arrMejorCromosoma, fMejorFitness = fnAlgoritmoGenetico()  # Ejecuta el algoritmo genético
    arrMiembros = np.where(arrMejorCromosoma == 1)[0]  # Obtiene los índices de los miembros de la coalición
    matPuntosCoalicion = dmatPosiciones[arrMiembros]  # Obtiene las coordenadas de los miembros
    if len(matPuntosCoalicion) > 2:
        hullFinal = ConvexHull(matPuntosCoalicion)
        arrVerticesRel = hullFinal.vertices
        arrVerticesOriginal = arrMiembros[arrVerticesRel]
    else:
        arrVerticesOriginal = []
    plt.figure(figsize=(6,6))
    plt.scatter(dmatPosiciones[:,0], dmatPosiciones[:,1], c='lightgray', label='Diputados')
    plt.scatter(matPuntosCoalicion[:,0], matPuntosCoalicion[:,1], c='blue', label='Miembro MWC')
    if len(matPuntosCoalicion) > 2:
        for simplex in hullFinal.simplices:
            plt.plot(matPuntosCoalicion[simplex, 0], matPuntosCoalicion[simplex, 1], 'r-')
        plt.scatter(matPuntosCoalicion[arrVerticesRel,0], matPuntosCoalicion[arrVerticesRel,1],
                    c='red', marker='o', edgecolors='k', label='Vértices del Hull')
    plt.xlabel('Dimensión económica (izquierda–derecha)')
    plt.ylabel('Dimensión social (progresismo–conservadurismo)')
    plt.title('Coalición Ganadora Mínima (MWC)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("coalicion_mwc.png")
    with open("MWC_resultados.csv", "w", newline='', encoding="utf-8") as fCSV:
        writer = csv.writer(fCSV)
        writer.writerow(["Index", "Nombre", "Partido", "X", "Y", "EsVerticeHull"])
        for idx in arrMiembros:
            nombre = dlstDiputados[idx].get('name', '')
            partido = dlstDiputados[idx].get('party_short_name', '')
            x = dmatPosiciones[idx,0]
            y = dmatPosiciones[idx,1]
            es_vertice = 1 if (len(matPuntosCoalicion) > 2 and idx in arrVerticesOriginal) else 0
            writer.writerow([idx, nombre, partido, f"{x:.4f}", f"{y:.4f}", es_vertice])
    print("\nResultados finales:")
    print(f"Fitness mínimo alcanzado: {fMejorFitness:.6f}")
    print("Imagen de la coalición guardada en 'coalicion_mwc.png'")
    print("Datos de la coalición guardados en 'MWC_resultados.csv'")

# Ejecutar si se llama directamente
if __name__ == "__main__":
    fnMain()
