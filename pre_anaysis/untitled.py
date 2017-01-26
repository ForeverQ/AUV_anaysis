import numpy as np

a = np.array([1,2,3])
b = np.array([1,2,3])

try:
    a.all() and c.all()
except NameError:
    print('NE')
else:
    print('Defined')