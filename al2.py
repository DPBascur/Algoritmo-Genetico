import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
import random
import json
import os
import matplotlib.pyplot as plt
import csv

# Parámetros globales
iDiputados = 431
iQuorum = 216
iPoblacionSize = 38 #100
iGeneraciones = 10000
fProbMutacion = 0.1700019 #0.02

# Carga de datos desde JSON
def fnCargarDatosJSON(strPath="VoteData.json"):
    try:
        fArchivo = open(strPath, 'r', encoding='utf-8')
    except FileNotFoundError:
        strBase = os.path.dirname(__file__)
        fArchivo = open(os.path.join(strBase, strPath), 'r', encoding='utf-8')
    dictData = json.load(fArchivo)
    fArchivo.close()
    lstDiputados = [v for v in dictData['rollcalls'][0]['votes']]
    iTotalDiputados = len(lstDiputados)
    iQuorumCalculado = iTotalDiputados // 2 + 1
    matPosiciones = np.array([[d['x'], d['y']] for d in lstDiputados])
    return lstDiputados, matPosiciones, iTotalDiputados, iQuorumCalculado

lstDiputados, matPosiciones, iDiputados, iQuorum = fnCargarDatosJSON()
matDistancias = squareform(pdist(matPosiciones, metric='euclidean'))

# Función de evaluación
def fnFitness(arrCromosoma):
    arrIdxMiembros = np.where(arrCromosoma == 1)[0]
    if len(arrIdxMiembros) < iQuorum:
        return 1e9
    if len(arrIdxMiembros) < 2:
        return 0.0
    matDistSub = matDistancias[np.ix_(arrIdxMiembros, arrIdxMiembros)]
    fSumaDistancias = matDistSub.sum() / 2.0
    return fSumaDistancias

# Inicialización de población
def fnInicializarPoblacion():
    lstPoblacion = []
    for _ in range(iPoblacionSize):
        arrCromosoma = np.zeros(iDiputados, dtype=int)
        arrSeleccionados = np.random.choice(iDiputados, iQuorum, replace=False)
        arrCromosoma[arrSeleccionados] = 1
        lstPoblacion.append(arrCromosoma)
    return np.array(lstPoblacion)

# Selección por torneo
def fnSeleccion(arrPoblacion, arrFitness):
    iIndex1, iIndex2 = random.sample(range(iPoblacionSize), 2)
    return arrPoblacion[iIndex1] if arrFitness[iIndex1] < arrFitness[iIndex2] else arrPoblacion[iIndex2]

# Cruza especializada
def fnCruzaMejorado(arrPadre1, arrPadre2):
    arrComun = np.logical_and(arrPadre1, arrPadre2)
    iComun = np.sum(arrComun)
    arrHijo1 = arrComun.astype(int)
    arrHijo2 = arrComun.astype(int)
    iFaltantes = iQuorum - iComun
    if iFaltantes > 0:
        arrDiferencia = np.logical_xor(arrPadre1, arrPadre2)
        idxOpciones = np.where(arrDiferencia == 1)[0]
        np.random.shuffle(idxOpciones)
        seleccionHijo1 = idxOpciones[:iFaltantes]
        seleccionHijo2 = idxOpciones[iFaltantes:2*iFaltantes] if len(idxOpciones) >= 2*iFaltantes else np.random.choice(idxOpciones, iFaltantes, replace=False)
        arrHijo1[seleccionHijo1] = 1
        arrHijo2[seleccionHijo2] = 1
    return arrHijo1, arrHijo2

# Mutación tipo swap con probabilidad

def fnMutarSwapProb(arrCromosoma):
    if np.random.rand() < fProbMutacion:
        arrMiembros = np.where(arrCromosoma == 1)[0]
        arrNoMiembros = np.where(arrCromosoma == 0)[0]
        if len(arrMiembros) > 0 and len(arrNoMiembros) > 0:
            iSalir = random.choice(arrMiembros)
            iEntrar = random.choice(arrNoMiembros)
            arrCromosoma[iSalir] = 0
            arrCromosoma[iEntrar] = 1
    return arrCromosoma

# Algoritmo genético principal
def fnAlgoritmoGenetico():
    arrPoblacion = fnInicializarPoblacion()
    fMejorFitness = 1e9
    arrMejorCromosoma = None
    gen_encontrado = None
    for iGen in range(iGeneraciones):
        arrFitnessPob = np.array([fnFitness(crom) for crom in arrPoblacion])
        iIdxMejor = np.argmin(arrFitnessPob)
        if arrFitnessPob[iIdxMejor] < fMejorFitness:
            fMejorFitness = arrFitnessPob[iIdxMejor]
            arrMejorCromosoma = arrPoblacion[iIdxMejor].copy()
        print(f"Generación {iGen+1}: Mejor fitness = {fMejorFitness:.6f} | Tamaño coalición = {np.sum(arrMejorCromosoma)}")
        if fMejorFitness <= 9690:
            gen_encontrado = iGen + 1
            print(f"\n¡Fitness objetivo 9690 o mejor alcanzado en la generación {gen_encontrado}!")
            break
        lstNuevaPoblacion = [arrMejorCromosoma]
        while len(lstNuevaPoblacion) < iPoblacionSize:
            arrPadre1 = fnSeleccion(arrPoblacion, arrFitnessPob)
            arrPadre2 = fnSeleccion(arrPoblacion, arrFitnessPob)
            arrHijo1, arrHijo2 = fnCruzaMejorado(arrPadre1, arrPadre2)
            arrHijo1 = fnMutarSwapProb(arrHijo1)
            arrHijo2 = fnMutarSwapProb(arrHijo2)
            lstNuevaPoblacion.extend([arrHijo1, arrHijo2])
        arrPoblacion = np.array(lstNuevaPoblacion[:iPoblacionSize])
    return arrMejorCromosoma, fMejorFitness, gen_encontrado

# Función principal
def fnMain():
    arrMejorCromosoma, fMejorFitness, gen_encontrado = fnAlgoritmoGenetico()
    arrMiembros = np.where(arrMejorCromosoma == 1)[0]
    matPuntosCoalicion = matPosiciones[arrMiembros]
    if len(matPuntosCoalicion) > 2:
        hullFinal = ConvexHull(matPuntosCoalicion)
        arrVerticesRel = hullFinal.vertices
        arrVerticesOriginal = arrMiembros[arrVerticesRel]
    else:
        arrVerticesOriginal = []
    plt.figure(figsize=(6,6))
    plt.scatter(matPosiciones[:,0], matPosiciones[:,1], c='lightgray', label='Diputados')
    plt.scatter(matPuntosCoalicion[:,0], matPuntosCoalicion[:,1], c='blue', label='Miembro MWC')
    if len(matPuntosCoalicion) > 2:
        for simplex in hullFinal.simplices:
            plt.plot(matPuntosCoalicion[simplex, 0], matPuntosCoalicion[simplex, 1], 'r-')
        plt.scatter(matPuntosCoalicion[arrVerticesRel,0], matPuntosCoalicion[arrVerticesRel,1],
                    c='red', marker='o', edgecolors='k', label='Vértices del Hull')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.title('Coalición Ganadora Mínima (MWC)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig("coalicion_mwc.png")
    with open("MWC_resultados.csv", "w", newline='', encoding="utf-8") as fCSV:
        writer = csv.writer(fCSV)
        writer.writerow(["Index", "Nombre", "Partido", "X", "Y", "EsVerticeHull"])
        for idx in arrMiembros:
            nombre = lstDiputados[idx].get('name', '')
            partido = lstDiputados[idx].get('party_short_name', '')
            x = matPosiciones[idx,0]
            y = matPosiciones[idx,1]
            es_vertice = 1 if (len(matPuntosCoalicion) > 2 and idx in arrVerticesOriginal) else 0
            writer.writerow([idx, nombre, partido, f"{x:.4f}", f"{y:.4f}", es_vertice])
    print("\nResultados finales:")
    print(f"Fitness mínimo alcanzado: {fMejorFitness:.6f}")
    if gen_encontrado:
        print(f"Fitness <= 9690 alcanzado en la generación: {gen_encontrado}")
    else:
        print("No se alcanzó el fitness objetivo en las generaciones dadas.")
    print("Imagen de la coalición guardada en 'coalicion_mwc.png'")
    print("Datos de la coalición guardados en 'MWC_resultados.csv'")

if __name__ == "__main__":
    fnMain()