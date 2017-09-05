import numpy as np
from random import randint
from random import uniform
from scipy.spatial.distance import euclidean
import time
from threading import Thread
from data import Data
import math

class Ant():
    ''' Initializes ant '''
    def __init__(self, x, y, radius, grid, its, alpha):
        s                   = grid.shape[0] // 2
        self.grid           = grid
        self.radius         = radius
        self.x              = x
        self.y              = y
        self.iterations     = its
        self.calc_r_()
        self.carrying       = False
        self.data           = None
        self.c              = self.radius*10
        self.max_step_size  = self.grid.shape[0] // 2 + 1
        self.alpha          = alpha

    ''' Calculates how many tiles an ant can see around itself '''
    def calc_r_(self):
        self.r_ = 1
        for i in range(self.radius):
            self.r_ = self.r_ + 2
        seen = self.neighbors(self.grid, self.x, self.y, n=self.r_)
        self.curr_rad = (seen.shape[0] **2) - 1

    def get_carrying(self):
        return self.carrying


    ''' Generates a random position within a random step size.
        The step size was introduced as means of speeding up
        the overall speed of the algorithm. '''
    def randpos(self):
        step_size = np.random.randint(1, self.max_step_size)
        #step_size = 1
        grid_shape = self.grid.shape[0]
        #step_size = 1
        x = self.x + np.random.randint(-1 * step_size ,1 * step_size +1)
        y = self.y + np.random.randint(-1 * step_size,1 * step_size +1)
        if x < 0: x = grid_shape + x
        if x >= grid_shape: x = x - grid_shape
        if y < 0: y = grid_shape + y
        if y >= grid_shape: y = y - grid_shape
        return x,y

    ''' Given a 2D-array, returns an nxn array whose 'center'
        element is arr[x,y] '''
    def neighbors(self, arr,x,y,n=3):
        arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
        return arr[:n,:n]

    ''' Picks up an object if the similarity function
        applied to the sigmoid function has a value
        greater than a random float between 0 and 1. '''
    def pick(self):
        seen = self.neighbors(self.grid, self.x, self.y, n=self.r_)

        fi = self.similarity(seen)
        #if fi >= 1.0: f = 1.0
        #else: f = 1/(fi**2)
        #print("sigmoid: " + str(self.sigmoid(self.c, fi)))
        sig = self.sigmoid(self.c, fi)
        f = 1 - sig
        rd = np.random.uniform(0.0, 1.0)
        #print("sig: " + str(sig) + "\npick: " + str(f) + "\nrd: " + str(rd) + "\n")

        if f >= rd:
            self.carrying = True
            self.data = self.grid[self.x, self.y]
            self.grid[self.x, self.y] = None
            return True
        return False

    ''' Drops an object if the similarity function
        applied to the sigmoid function has a value
        greater than a random float between 0 and 1. '''
    def drop(self):
        seen = self.neighbors(self.grid, self.x, self.y, n=self.r_)

        fi = self.similarity(seen)
        # if fi >= 1.0: f = 1.0
        # else: f = (fi**4)
        f = self.sigmoid(self.c, fi)

        rd = np.random.uniform(0.0, 1.0)
        #print("drop: " + str(f) + "\nrd: " + str(rd) + "\n")


        if f >= rd:
            self.carrying = False
            self.grid[self.x, self.y] = self.data
            self.data = None
            return True
        return False

    ''' Main function used to move the agent on the
        grid. If #iterations has reached 0 and the
        agent is carrying an object, moves the agent
        until the item is dropped. '''
    def run(self):
        self.move()
        if self.iterations <= 0 and self.carrying:
            while self.carrying:
                self.move()

    def move(self):
        grid = self.grid
        x, y = self.x, self.y

        if grid[x,y] == None:
            if self.carrying:
                self.drop()
        elif grid[x,y] != None:
            if not self.carrying:
                self.pick()

        self.x, self.y = self.randpos()
        self.iterations -= 1

    def similarity(self, seen):
        s = 0
        shape = seen.shape[0]
        if self.carrying:
            data = self.data.get_attribute()
        else:
            data = self.grid[self.x, self.y].get_attribute()

        for i in range(shape):
            for j in range(shape):
                ret = 0
                if seen[i,j] != None:
                    ret = 1 - (euclidean(data,
                            seen[i,j].get_attribute()))/(self.alpha)
                    s += ret

        fi = s/(self.r_**2)
        if fi > 0: return fi
        else: return fi

    def sigmoid(self, c, x):
        return ((1-np.exp(-(c*x)))/(1+np.exp(-(c*x))))
