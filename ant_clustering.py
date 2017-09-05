import pygame
import numpy as np
import sys
from ant import Ant
from random import randint
from scipy.spatial.distance import euclidean
import threading
import time
from pygame.locals import *
from data import Data
flags = DOUBLEBUF

# grid 100x100
# 5k itens
# 50 agentes
# raio minimo 1, 5 e 10
# iterações: max 5mi

class AntClustering():
    def __init__(self, grid=100,
                 rad=2,
                 dead_ants=5000,
                 antnum=50,
                 iterations=5*10**6,
                 fname='datasets/400.txt',
                 alpha=0,
                 sleep=False,
                 dsize=500):

        self.size       = grid          # Grid size
        self.rad        = rad           # How far can ants see?
        self.antnum     = antnum        # Number of workers
        self.iterations = iterations
        self.workers    = list()        # Worker ant list
        self.d_size     = dsize           # Display size
        self.dtypes     = [1,2,3,4]     # Datatypes
        self.data       = self._load_data(fname)
        self.sleep      = sleep

        ''' Calculates alpha if not provided '''
        if alpha == 0:
            self.alpha  = self.calc_alpha()
        else:
            self.alpha  = alpha
        print("alpha: " + str(self.alpha))

        ''' Generates grid '''
        self.grid = np.empty((self.size, self.size), dtype=np.object)
        self._distribute_data(self.grid, self.data)
        print(self.grid)
        #print(self.calc_alpha())

        ''' Initializes ant agents '''
        self._create_ants(self.antnum,
                          self.rad,
                          self.grid,
                          self.iterations,
                          self.alpha)

    ''' Loads dataset and returns a list of Data items '''
    def _load_data(self, fname):
        info = np.loadtxt(fname)
        ret = list()
        for d in info:
            n_data = Data(d[0:-1], d[-1]*5%255) # *10%255 only for coloring purposes
            ret.append(n_data)
        return ret

    ''' Calculates alpha value used in similarity function '''
    def calc_alpha(self):
        s = 0
        for d1 in self.data:
            for d2 in self.data:
                s += euclidean(d1.get_attribute(), d2.get_attribute())
        return s/(len(self.data)**2)

    ''' Randomly distributes data over the grid '''
    def _distribute_data(self, grid, data):
        for d in data:
            #print(d)
            i = np.random.randint(0, self.size)
            j = np.random.randint(0, self.size)
            while grid[i,j] is not None:
                i = np.random.randint(0, self.size)
                j = np.random.randint(0, self.size)
            grid[i,j] = d

    ''' Create ant agents and appends to worker list '''
    def _create_ants(self, antnum, rad, grid, its, alpha):
        for i in range(antnum):
            x = np.random.randint(0, self.size-1)
            y = np.random.randint(0, self.size-1)
            ant = Ant(x, y, rad, grid, its, alpha)
            self.workers.append(ant)

    ''' Starts sequential execution '''
    def _start_seq(self):
        if self.sleep: time.sleep(5)
        for i in range(self.iterations // self.antnum):
            for ant in self.workers:
                ant.run()
        l = list()
        for ant in self.workers:
            l.append(ant.get_carrying())
        print(l)

    ''' Converts Data matrix into a matrix of ints in order
        to display it on pygame screen '''
    def _get_dmatrix(self):
        ret = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i,j] != None and type(self.grid[i,j]) == Data:
                    data = self.grid[i,j]
                    ret[i,j] = data.get_group()
                else:
                    ret[i,j] = self.grid[i,j]
        return ret

    ''' Main loop '''
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.d_size,self.d_size))
        screen.set_alpha(None)
        print(np.count_nonzero(self.grid != None))
        t = threading.Thread(target=self._start_seq)
        t.daemon=True
        t.start()
        #t.join()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            int_arr = self._get_dmatrix()
            surface = pygame.surfarray.make_surface(int_arr)
            newsurface = pygame.transform.scale(surface,
                                                (self.d_size,
                                                self.d_size))
            screen.blit(newsurface, (0,0))
            pygame.display.flip()

if __name__ == "__main__":
    antcluster = AntClustering(rad=5, grid=100, antnum=50)
    antcluster.run()
