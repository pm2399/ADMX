import numpy as np
import matplotlib.pyplot as plt

filenames = [
 	"Q_reflection_experiment.CSV",
 	"Q_experiment.CSV"
]
A = np.ndarray(shape = (7,2))
cavity_length_experiment = np.array([17,17.5,18,18.5,19,19.5,20])
for index in range(0,len(filenames)):
	A[:,index] = np.genfromtxt(filenames[index])
	#plt.plot(cavity_length_experiment,A)
	#plt.yscale('log')
print(A)
A = np.absolute(A)
plt.plot(cavity_length_experiment,A[:,0],'b-',label = 'reflection')
plt.plot(cavity_length_experiment,A[:,1],'r--',label = 'transmission')
plt.xlabel('cavity length')
plt.ylabel('quality factor(log)')
plt.legend()
plt.yscale('log')
#plt.show()
plt.savefig("q_reflection_vs_transmission_logscale.pdf",bbox_inches = 'tight')