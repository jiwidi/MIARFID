# -*- coding: utf-8 -*-
import copy
import numpy as np
from sys import *
from distances import get_distance_matrix
from matplotlib.pyplot import matshow, show, cm, plot
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from scipy.signal import argrelextrema


def lee_fichero(fichero):
    matriz = []
    fichero = open(fichero, "r")
    lineas = fichero.readlines()
    matriz = [linea.split() for linea in lineas]
    fichero.close()
    return np.array(matriz).astype(np.float)


def stdw(distancias):
    matrix = np.zeros(shape=(len(npmatrix1), len(npmatrix2), 4))
    # Calculamos T i,0
    # Calculamos T 0,j
    for j in range(len(npmatrix2)):
        matrix[0][j] = [distancias[0, j], 1, j, distancias[0, j]]

    for i in range(1, len(npmatrix1)):
        matrix[i][0] = [
            distancias[i, 0] + matrix[i - 1][0][0],
            i,
            0,
            distancias[i, 0] + matrix[i - 1][0][0] / i,
        ]

    # Calculamos T i,j
    for j in range(1, len(npmatrix2)):
        for i in range(1, len(npmatrix1)):
            aux1 = (matrix[i - 1][j][0] + distancias[i][j]) / (matrix[i - 1][j][1] + 1)
            aux2 = (matrix[i][j - 1][0] + distancias[i][j]) / (matrix[i][j - 1][1] + 1)
            aux3 = (matrix[i - 1][j - 1][0] + distancias[i][j]) / (
                matrix[i - 1][j - 1][1] + 1
            )

            aux = min(aux1, aux2, aux3)

            if aux == aux1:
                matrix[i][j][0] = matrix[i - 1][j][0] + distancias[i][j]
                matrix[i][j][2] = matrix[i - 1][j][2]
            elif aux == aux2:
                matrix[i][j][0] = matrix[i][j - 1][0] + distancias[i][j]
                matrix[i][j][2] = matrix[i][j - 1][2]
            else:
                matrix[i][j][0] = matrix[i - 1][j - 1][0] + distancias[i][j]
                matrix[i][j][2] = matrix[i - 1][j - 1][2]

            matrix[i][j][1] = i
            matrix[i][j][3] = matrix[i][j][0] / matrix[i][j][1]

    last_rows = np.array(
        [matrix[len(npmatrix1) - 1][j][3] for j in range(1, len(npmatrix2))]
    )
    starts = np.array(
        [matrix[len(npmatrix1) - 1][j][2] for j in range(1, len(npmatrix2))]
    )
    return last_rows, starts


if __name__ == "__main__":
    npmatrix1 = lee_fichero("mfc_queries/SEG-0032.mfc.raw")
    npmatrix2 = lee_fichero("largo250000.mfc.raw")

    print("Calculando distancias", end="\r")
    distancias = get_distance_matrix(npmatrix1, npmatrix2, "cos")
    print("Calculando distancias ... DONE")

    print("Calculando stdw", end="\r")
    last_row, starts = stdw(distancias)
    print("Calculando stdw ... DONE")

    ventana = 100
    print()
    print(f"Encontrando minimos locales", end="\r")
    min_index = argrelextrema(last_row, np.less)[0]
    minimos = np.take(last_row, min_index).tolist()
    starts = np.take(starts, min_index).tolist()

    _, starts = zip(*sorted(zip(minimos, starts)))
    minimos, min_index = zip(*sorted(zip(minimos, min_index)))

    # minimos_result = [[starts[i], min_index[i], minimos[i]] for i in range(10)]
    visited_starts = []
    minimos_result = []
    for i in range(len(minimos)):
        if starts[i] not in visited_starts:
            minimos_result.append([starts[i], min_index[i], minimos[i]])
            visited_starts.append(starts[i])
        if len(visited_starts) == 10:
            break

    print("Encontrando minimos ... DONE")
    print("TOP 10")
    print(
        tabulate(
            minimos_result,
            headers=["start-time", "finish-time", "coste"],
            tablefmt="orgtbl",
        )
    )
