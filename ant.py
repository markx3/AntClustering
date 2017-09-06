import numpy as np
from scipy.spatial.distance import euclidean
import time
from threading import Thread
from data import Data

class Ant():
    ''' Initializes ant '''
    def __init__(self, x, y, radius, grid, its, alpha):
        s                   = grid.shape[0] // 2
        self.grid           = grid
        self.radius         = radius
        self.x              = x
        self.y              = y
        self.iterations     = its
        self._calc_r_()
        self.carrying       = False
        self.data           = None
        self.c              = self.radius*10
        self.max_step_size  = self.grid.shape[0] // 2 + 1
        self.alpha          = alpha

    ''' Generates a random position within a random step size.
        The step size was introduced as means of speeding up
        the algorithm. '''
    def _randpos(self):
        step_size = np.random.randint(1, self.max_step_size)
        grid_shape = self.grid.shape[0]
        x = self.x + np.random.randint(-1 * step_size ,1 * step_size +1)
        y = self.y + np.random.randint(-1 * step_size,1 * step_size +1)
        if x < 0: x = grid_shape + x
        if x >= grid_shape: x = x - grid_shape
        if y < 0: y = grid_shape + y
        if y >= grid_shape: y = y - grid_shape
        return x,y

    ''' Given a 2D-array, returns an nxn array whose 'center'
        element is arr[x,y] '''
    def _neighbors(self, arr,x,y,n=3):
        arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
        return arr[:n,:n]

    ''' picks up an object if the _avg_similarity function
        applied to the _sigmoid function has a value
        greater than a random float between 0 and 1. '''
    def _pick(self):
        seen = self._neighbors(self.grid, self.x, self.y, n=self.r_)
        fi = self._avg_similarity(seen)
        sig = self._sigmoid(self.c, fi)
        f = 1 - sig
        rd = np.random.uniform(0.0, 1.0)
        #print("sig: " + str(sig) + "\npick: " + str(f) + "\nrd: " + str(rd) + "\n")

        if f >= rd:
            self.carrying = True
            self.data = self.grid[self.x, self.y]
            self.grid[self.x, self.y] = None
            return True
        return False

    ''' drops an object if the _avg_similarity function
        applied to the _sigmoid function has a value
        greater than a random float between 0 and 1. '''
    def _drop(self):
        seen = self._neighbors(self.grid, self.x, self.y, n=self.r_)
        fi = self._avg_similarity(seen)
        f = self._sigmoid(self.c, fi)
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
        until the item is _dropped. '''
    def run(self):
        self._move()
        if self.iterations <= 0 and self.carrying:
            while self.carrying:
                self._move()

    ''' Moves the ant around the grid, whilst checking
        if it's above an object or not. '''
    def _move(self):
        grid = self.grid
        x, y = self.x, self.y

        if grid[x,y] == None:
            if self.carrying:
                self._drop()
        elif grid[x,y] != None:
            if not self.carrying:
                self._pick()

        self.x, self.y = self._randpos()
        self.iterations -= 1

    ''' Calculates the average similarity between an
        object and the objects around the agent '''
    def _avg_similarity(self, seen):
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
                            seen[i,j].get_attribute()))/((self.alpha))
                    s += ret

        fi = s/(self.r_**2)
        if fi > 0: return fi
        else: return 0

    ''' Normalizes the _avg_similarity function '''
    def _sigmoid(self, c, x):
        return ((1-np.exp(-(c*x)))/(1+np.exp(-(c*x))))

    ''' Calculates how many tiles an ant can see around itself '''
    def _calc_r_(self):
        self.r_ = 1
        for i in range(self.radius):
            self.r_ = self.r_ + 2

    def _get_carrying(self):
        return self.carrying
