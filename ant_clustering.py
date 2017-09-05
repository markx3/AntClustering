import pygame
import numpy as np
import sys
from ant import Ant
from random import randint
import threading
import time
from pygame.locals import *
flags = DOUBLEBUF

# grid 100x100
# 5k itens
# 50 agentes
# raio minimo 1, 5 e 10
# iterações: max 5mi

class AntClustering():
    def __init__(self, grid=100,
                 rad=1,
                 dead_ants=5000,
                 antnum=50,
                 iterations=5*10**6):

        self.size       = grid          # Grid size
        self.rad        = rad           # How far can ants see?
        self.antnum     = antnum        # Number of workers
        self.iterations = iterations
        self.workers    = list()        # Worker ant list
        self.d_size     = 500           # Display size
        self.dtypes     = [120]         # Datatypes
        self.data       = [120]

        ''' Generates grid '''
        self.grid = np.zeros((self.size, self.size))
        self._distribute_data(self.grid, dead_ants)

        ''' Initializes ant agents '''
        self._create_ants(self.antnum,
                          self.rad,
                          self.grid,
                          iterations // antnum,
                          self.dtypes)

    ''' Randomly distributes data over the grid '''
    def _distribute_data(self, grid, data):
        for _ in range(data):
            i = np.random.randint(0, self.size)
            j = np.random.randint(0, self.size)
            while grid[i,j] != 0:
                i = np.random.randint(0, self.size)
                j = np.random.randint(0, self.size)
            grid[i,j] = self.data[0]

    def _create_ants(self, antnum, rad, grid, its, dtypes):
        for i in range(antnum):
            x = np.random.randint(0, self.size-1)
            y = np.random.randint(0, self.size-1)
            ant = Ant(x, y, rad, grid, its, dtypes)
            self.workers.append(ant)

    def _start_seq(self):
        for _ in range(self.iterations // self.antnum):
            for ant in self.workers:
                ant.run()
        l = list()
        for ant in self.workers:
            l.append(ant.get_carrying())
        print(l)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.d_size,self.d_size))
        screen.set_alpha(None)
        print(np.count_nonzero(self.grid == 120))
        t = threading.Thread(target=self._start_seq)
        t.daemon=True
        t.start()
        #t.join()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            surface = pygame.surfarray.make_surface(self.grid)
            newsurface = pygame.transform.scale(surface,
                                                (self.d_size,
                                                self.d_size))
            screen.blit(newsurface, (0,0))
            pygame.display.flip()

if __name__ == "__main__":
    antcluster = AntClustering(rad=5)
    antcluster.run()
