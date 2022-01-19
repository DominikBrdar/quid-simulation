from array import array
from timeit import timeit
from xml import dom
from numba import cuda
import cupy as cp


from main2 import *


creation(R_NUM.value, G_NUM.value, B_NUM.value, Y_NUM.value, ARR_X.value, ARR_Y.value)

#size = quid_counter(listOfQuids)
x = np.gcd(int(np.ceil(np.sqrt(MAX_QUIDS.value))), int(np.ceil(MAX_QUIDS.value)))
y = int(MAX_QUIDS.value / x)

#listOfQuidsMat = listOfQuids.reshape(x, y)


posx = cp.empty((x,y))
for i in range(x):
    for j in range(y):
        if listOfQuids[i+j]:
            posx[i][j] = listOfQuids[i+j].pos[0]
        else:
            posx[i][j] = -100
        

dirx = cp.empty((x,y))
for i in range(x):
    for j in range(y):
        if listOfQuids[i+j]:
            dirx[i][j] = listOfQuids[i+j].dir[0]
        else:
            dirx[i][j] = 0

posx = posx + dirx

print(posx)