import numpy as np

def sigfigs(number):
	numbersign = np.sign(number)
	numsign = numbersign[0]
	number = number*numsign
	return number

x = input()
x = sigfigs(x)
print(x)


