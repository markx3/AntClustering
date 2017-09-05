import numpy as np
from random import randint
from random import uniform
from numba import jit
from scipy.spatial.distance import euclidean
import time
from threading import Thread

class Ant():
    def __init__(self, x, y, radius, grid, its, dtypes):
        s = int(grid.shape[0] / 2)
        self.grid = grid
        self.radius = radius
        self.x = x
        self.y = y
        self.dtypes = dtypes
        self.iterations = its
        self.calc_r_()
        self.carrying = False
        self.data = 0

    def calc_r_(self):
        self.r_ = 1
        for i in range(self.radius):
            self.r_ = self.r_ + 2
        seen = self.neighbors(self.grid, self.x, self.y, n=self.r_)
        self.curr_rad = (seen.shape[0] **2) - 1

    def get_carrying(self):
        return self.carrying

    def randpos(self):
        step_size = np.random.randint(1, 25)
        grid_shape = self.grid.shape[0]
        x = self.x + np.random.randint(-1 * step_size ,1 * step_size +1)
        y = self.y + np.random.randint(-1 * step_size,1 * step_size +1)
        if x < 0: x = grid_shape + x
        if x >= grid_shape: x = x - grid_shape
        if y < 0: y = grid_shape + y
        if y >= grid_shape: y = y - grid_shape
        return x,y

    def neighbors(self, arr,x,y,n=3):
        #Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]
        arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
        return arr[:n,:n]

    def pick_or_drop(self):
        if self.data == 0:
            choice = self.dtypes[np.random.randint(0, len(self.dtypes))]
        else: choice = self.data
        seen = self.neighbors(self.grid, self.x, self.y, n=self.r_)
        deadcount = np.count_nonzero(seen == choice)
        grid = self.grid
        if grid[self.x, self.y] == 0 and self.carrying:
            ratio = deadcount/self.curr_rad
            rd = np.random.uniform(0.2, 1.0)
            if ratio >= rd:
                self.carrying = False
                grid[self.x, self.y] = self.data
                self.data = 0
            return
        if grid[self.x, self.y] == choice and not self.carrying:
            ratio = deadcount/self.curr_rad
            rd = np.random.uniform(0.2, 1.0)
            # fi = self.similarity(seen, deadcount)
            # if fi <= 1.0: f = 1.0
            # else: f = 1/(fi**2)
            # rd = uniform(0.0, 1.0)
            #print("pick: " + str(f) + "\nrd: " + str(rd))
            #rd = 0.5
            if ratio < rd:
                self.carrying = True
                self.data = grid[self.x, self.y]
                grid[self.x, self.y] = 0
            return

    def run(self):
        self.move()
        if self.iterations <= 0 and self.carrying:
            while self.carrying:
                self.move()

    def move(self):
        grid = self.grid
        x, y = self.x, self.y

        if grid[x, y] not in self.dtypes: grid[x, y] = 0
        self.pick_or_drop()
        x, y = self.randpos()
        if grid[x, y] not in self.dtypes:
            if not self.carrying: grid[x, y] = 50
            else: grid[x, y] = 20
        self.iterations -= 1
        self.x, self.y = x, y

    def similarity(self, seen, datacount):
        alpha = 100
        s = 0
        grid = self.grid
        shape = int(seen.shape[0])
        x,y = int(shape/2), int(shape/2)
        for i in range(shape):
            for j in range(shape):
                ret = 0
                if seen[i,j] != 0:
                    ret = ((1-(euclidean(self.data, seen[i,j]))/alpha))
                #if i == x and j == y: ret += 0
                #if ret > 0: s += ret
                #if ret > 0: s += ret
                #print(s)
        #print(seen.shape[0]**2)
        #print(self.r_**2)
        fi = s/(self.r_**4)
        #print(fi)
        if fi > 0: return fi
        else: return 0
