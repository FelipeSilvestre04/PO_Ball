import numpy as np
# TSP - Caxeiro Viajante

# Cidade1, Cidade2, Cidade3 ... CidadeN


def tsp(n):
    cidades_matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cidades_matriz[i, j] = np.random.randint(1, 100)
    return cidades_matriz


def GA(matriz_distancia, n_populacao, n_iteracoes):

    

    for i in range(n_iteracoes):
        
        
    
    