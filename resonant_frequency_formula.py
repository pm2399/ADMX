import math
def resonant_frequency(q,m,n,d):
	return 2*(q+1)*((3*10**8)/(4*(d/100)))+(((3*10**8)/(4*(d/100)))/math.pi)*(1+m+n)*math.acos(1-((2*d)/34.59))

#def fundamental_frequency(d):
	# return (3*10**8)/(4*(d/100))

#fundamental_f = fundamental_frequency(17.8)
calculator = resonant_frequency(19,0,0,19)/(10**9)
print(calculator)
