import numpy as np

info = np.loadtxt('breast-cancer-wisconsin.txt', delimiter=',')
np.savetxt('breast-cancer-wisconsin2.txt', info[:,1:], delimiter="\t",
            fmt="%.5f", )
