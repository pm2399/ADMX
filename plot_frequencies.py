import numpy as np
import matplotlib.pyplot as plt

M = np.genfromtxt('measured_resonant_frequencies.csv',delimiter=',',skip_header=1)

L= M[:,0]
Fp = M[:,1]
Fm = M[:,2]
diff = Fp-Fm

plt.figure()
plt.plot(L,Fp, label='Predicted Resonance')
plt.plot(L,Fm, label='Measured Resonance')
plt.legend(loc='best')
plt.xlabel('Cavity Length [cm]')
plt.ylabel('Resonant Frequency [GHz]')
#plt.savefig('measured_and_predicted_frequencies_1.pdf')
plt.show()
plt.figure()
plt.plot(L, diff)
plt.xlabel('Cavity Length [cm]')
plt.ylabel('Predicted - Measured Frequency [Ghz]')
#plt.savefig('measured_and_predicted_frequencies_2.pdf')
#plt.close()
plt.show()
