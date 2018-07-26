from sympy import Symbol, I, Abs
import numpy as np
import matplotlib.pyplot as plt
x = Symbol('x')
y = Symbol('y')
new_func = abs(1-x+2*I*x*y+4*y**2)
funct = pow(1-x+2*I*x*y+4*y**2,2)
funct = funct.expand()
print(funct)
 
