import numpy as np
from scipy.linalg import svd
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[8,9,10],[11,12,13]])
print("\nFirst Matrix:",a)
print("\nSecond Matrix:",b)
print("\nMatrix Addition:",np.add(a,b))
print("\nMatrix Subtraction:",np.subtract(a,b))
print("\nMatrix Multiplication:",np.multiply(a,b))
print("\nMatrix Division:",np.divide(a,b))
U,s,VT=svd(a)
print("\nDecompose:",s)
print("\nInverse:",U)
print("\nMatrix Transpose:",VT)





















