# -*- coding: utf-8 -*-
import copy
import numpy as np
from sys import *
from distances import get_distance_matrix
from matplotlib.pyplot import matshow, show, cm, plot
import matplotlib.pyplot as plt
import math
from tabulate import tabulate


def lee_fichero(fichero):
    matriz = []
    fichero = open(fichero, "r")
    lineas = fichero.readlines()
    matriz = [linea.split() for linea in lineas]
    fichero.close()
    return np.array(matriz).astype(np.float)


def stdw(distancias):
    matrix = np.zeros(shape=(len(query), len(audio), 3))
    # Calculamos T i,0
    for i in range(1, len(query)):
        matrix[i][0] = [
            distancias[i, 0] + matrix[i - 1][0][0],
            matrix[i - 1][0][1] + 1,
            matrix[i - 1][0][2],
        ]
    # Calculamos T 0,j
    for j in range(len(audio)):
        matrix[0][j] = [distancias[0, j], 0, j]
    # Calculamos T i,j
    result = [0, 0, 0]
    for j in range(1, len(audio)):
        for i in range(1, len(query)):
            aux1 = (matrix[i - 1][j - 1][0] + distancias[i, j]) / (
                matrix[i - 1][j - 1][1] + 1
            )
            aux2 = (matrix[i - 1][j][0] + distancias[i, j]) / \
                (matrix[i - 1][j][1] + 1)
            aux3 = (matrix[i][j - 1][0] + distancias[i, j]) / \
                (matrix[i][j - 1][1] + 1)
            options = [[aux1, i - 1, j - 1],
                       [aux2, i - 1, j], [aux3, i, j - 1]]
            aux = [options[0][0], options[1][0], options[2][0]]
            index_min = aux.index(min(aux))
            result[0] = (
                matrix[options[index_min][1]][options[index_min][2]][0]
                + distancias[i, j]
            )
            result[1] = matrix[options[index_min]
                               [1]][options[index_min][2]][1] + 1
            result[2] = matrix[options[index_min][1]][options[index_min][2]][2]

            matrix[i][j] = result

    last_rows = [matrix[len(query) - 1][j] for j in range(len(audio))]
    return last_rows


if __name__ == "__main__":
    query = lee_fichero("mfc_queries/SEG-0062.mfc.raw")
    audio = lee_fichero("largo250000.mfc.raw")

    print("Calculando distancias", end="\r")
    distancias = get_distance_matrix(query, audio, "cos")
    print("Calculando distancias ... DONE")

    print("Calculando stdw", end="\r")
    last_row = stdw(distancias)
    print("Calculando stdw ... DONE")

    print("Encontrando minimos", end="\r")
    minimos = []
    audio_length = int(round(len(audio)))
    ventana = 100
    for index in range(0, audio_length, ventana):
        minimun_cost = math.inf
        minimo = None
        for x in range(index + 1, index + ventana):
            if last_row[x][0] < minimun_cost:
                minimun_cost = last_row[x][0]
                minimo = last_row[x]
        minimos.append(minimo)

    minimos.sort(key=lambda x: x[0])
    minimos_result = [
        [
            str(minimos[i][2] / 100),
            str((minimos[i][2] + minimos[i][1]) / 100),
            str(minimos[i][0]),
        ]
        for i in range(10)
    ]
    print("Encontrando minimos ... DONE")
    print("TOP 10")
    print(
        tabulate(
            minimos_result,
            headers=["start-time", "finish-time", "coste"],
            tablefmt="orgtbl",
        )
    )
