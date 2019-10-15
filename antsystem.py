import numpy as np
import math
import time
from random import *


class AntColonySystem:
    numAnts = 10 # número de formigas
    numIterations = 100 # número de caminhos que cada formiga fará
    a = 0.1  # alpha
    b = 2  # beta
    q0 = 0.9

    def __init__(self, size, cityLocations):
        self.size = size
        self.cityLocations = cityLocations
        self.costs = self.createCostMatrix(cityLocations)
        self.bestTour = None
        self.bestTourLength = math.inf
        self.tau = 1 / (self.lengthNearestNeighbour() * self.size)
        self.pheromone = self.tau * np.ones((self.size, self.size))

    def findSolution(self):
        T1 = time.perf_counter()
        location = np.zeros(self.numAnts, np.int32)
        startingpoint = np.zeros(self.numAnts, np.int32)

        for ant in range(self.numAnts):
            startingpoint[ant] = location[ant] = randint(0, self.size - 1)

        for i in range(self.numIterations):
            visited = np.zeros((self.numAnts, self.size), dtype=bool)
            tours = [np.zeros((self.size, self.size), np.int8) for ant in range(self.numAnts)]
            distances = np.zeros(self.numAnts)
            for ant in range(self.numAnts):
                visited[ant][location[ant]] = True

            for step in range(self.size):
                for ant in range(self.numAnts):
                    current = location[ant]
                    if step != self.size - 1:
                        next = self.nextCity(ant, location[ant], visited[ant])
                    else:
                        next = startingpoint[ant]
                    location[ant] = next
                    visited[ant][next] = True
                    tours[ant][current][next] = tours[ant][next][current] = 1
                    distances[ant] += self.costs[current][next]
                    self.localTrailUpdate(current, next)
            shortestLength = min(distances)
            if shortestLength < self.bestTourLength:
                self.bestTourLength = shortestLength
                self.bestTour = tours[np.argmin(distances)]
            self.globalTrailUpdate()
        T2 = time.perf_counter()
        print('Melhor percurso encontrado: ', self.bestTourList())
        print('Tem um tamanho de: ', self.bestTourLength )
        print('Encontrado em ', T2 - T1, ' segundos')

    # cria uma matriz de custos
    def createCostMatrix(self, cityLocations):
        # cria uma matriz quadrada size x size, preenchida com zeros
        result = np.zeros((self.size, self.size))
        # preenche toda a matriz com as distâncias entre os vértices
        for i in range(self.size):
            for j in range(self.size):
                result[i][j] = self.distance(i, j)
        return result

    # define a distância euclidiana entre duas cidades i e j
    def distance(self, i, j):
        return math.sqrt(
            math.pow(self.cityLocations[j][0]-self.cityLocations[i][0], 2) +
            math.pow(self.cityLocations[j][1]-self.cityLocations[i][1], 2))

    # retorna a cidade mais próxima e não-visitada
    def closestNotVisited(self, loc, visited):
        minimum = math.inf
        result = None
        for city in range(self.size):
            if (not visited[city]) and (self.costs[loc][city] < minimum):
                minimum = self.costs[loc][city]
                result = city
        return result

    # atualiza trilha  de feromônio entre i e j
    def localTrailUpdate(self, i, j):
        self.pheromone[j][i] = self.pheromone[i][j] = (1-self.a) * self.pheromone[i][j] + self.a * self.tau

    # atualiza feromônio de todas as arestas
    def globalTrailUpdate(self):
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.pheromone[i][j] = self.pheromone[j][i] = (1-self.a)*self.pheromone[i][j] + self.a * self.bestTour[i][j] / self.bestTourLength

    # retorna a próxima cidade que a formiga deve visitar
    def nextCity(self, ant, loc, visited):
        result = None
        q = np.random.random_sample()
        if q <= self.q0:
            max = -math.inf
            for city in range(self.size):
                if not visited[city]:
                    f = self.attraction(loc, city)
                    if f > max:
                        max = f
                        result = city
            if max != 0:
                return result
            else:
                return self.closestNotVisited(loc, visited)
        else:
            sum = 0
            for city in range(self.size):
                if not visited[city]:
                    sum += self.attraction(loc, city)
            if sum == 0:
                return self.closestNotVisited(loc, visited)
            else:
                R = np.random.random_sample()
                s = 0
                for city in range(self.size):
                    if not visited[city]:
                        s += self.attraction(loc, city) / sum
                        if s > R:
                            return city

    # retorna o nível de atração da aresta i,j
    def attraction(self, i, j):
        if i != j:
            return self.pheromone[i][j] / (math.pow(self.costs[i][j], self.b))
        else:
            return 0

    # retorna o custo toal do percurso, escolhendo sempre o vizinho mais próximo
    def lengthNearestNeighbour(self):
        # define o vértice inicial como um aleatório entre todos
        start = randint(0, self.size-1)
        # define o atual como o inicial
        current = start
        # retorna um array de visitados, todos como false
        visited = np.zeros(self.size, dtype=bool)
        # inicia o percurso já com o vértice atual (inicial)
        tour = [current]
        # inicia tamanho do percurso como 0
        length = 0
        # para cada vértice na posição i
        for i in range(self.size-1):
            # seta o vértice como visitado
            visited[current] = True
            # marca o custo mínimo como infinito
            minimum = math.inf
            # marca o mais próximo como indefinido
            closest = None
            # para cada vértice j, diferente de i
            for i in range(self.size):
                # se não foi visitado e o custo é menos do que o mínimo
                if (not visited[i]) and (self.costs[current][i] < minimum):
                    # mínimo será a distancia (custo) do vértice i para j
                    minimum = self.costs[current][i]
                    # o mais próximo será este vértice
                    closest = i
            # insere o vértice mais próximo no percurso
            tour.append(closest)
            # soma o custo do mínimo ao custo total do percurso
            length += minimum
            # marca o vértice mais próximo como o atual
            current = closest
        # no fim, o caixeiro volta para o ponto de partida
        tour.append(start)
        # e é acrescentado ao custo total, o custo do penúltimo vértice para o vértice inicial e retona o custo total
        length += self.costs[current][start]
        return length

    # Retorna o melhor passeio como uma lista de cidades na ordem em que são visitadas
    def bestTourList(self):
        current = 0
        previous = 0
        tour = [0]
        for i in range(self.size):
            next = 0
            while (self.bestTour[current][next] == 0) or (previous == next):
                next += 1
            tour.append(next)
            previous = current
            current = next
        return tour





