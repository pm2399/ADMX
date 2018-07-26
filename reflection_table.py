import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filenames1 = [
	"Q_reflection_thin_mirror.CSV",
	"Q_reflection_experiment.CSV",
]


filenames5 = [
	"resonant_frequencies_reflection_thin_mirror.CSV",
	"resonant_frequencies_reflection_experiment.CSV"
]

filenames6 = [
	"background_level_reflection_thin_mirror.CSV",
	"background_level_reflection_experiment.CSV"
]
cavity_length_experiment = np.array([17,17.5,18,18.5,19,19.5,20])
cavity_length_simulation = np.array([17,17.5,18,18.5,19,19.5,20.26])
final_table = np.ndarray(shape = (25,3))
table_q_loaded = np.ndarray(shape = (7,3))
for index in range(0, len(filenames1)):
	A = np.genfromtxt(filenames1[index])
	table_q_loaded[:,index] = A
table_q_loaded = np.absolute(table_q_loaded)
#np.savetxt("sd.CSV",table,delimiter = ",")
table_resonant_frequencies = np.ndarray(shape = (7,3))
for index in range(0, len(filenames5)):
	A = np.genfromtxt(filenames5[index])
	table_resonant_frequencies[:,index] = A
table_resonant_frequencies = np.absolute(table_resonant_frequencies)

table_background_level = np.ndarray(shape = (7,3))
for index in range(0, len(filenames6)):
	A = np.genfromtxt(filenames6[index])
	table_background_level[:,index] = A

final_table[0:7,:] = table_q_loaded
final_table[9:16,:] = table_resonant_frequencies
final_table[18:25,:] = table_background_level


resfre = np.ndarray(shape = (7,5))
resfre[:,0:3] = table_resonant_frequencies
resfre[:,3] = [17.8719,17.3613,16.8790,16.4228,15.9907,15.5806,15.1191]
resfre[:,4] = resfre[:,2]-resfre[:,3]
#print(resfre)
#np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})
#print(final_table)
#table = pd.DataFrame(final_table)
#table.to_csv("table.CSV")
#table_q_loaded = np.log(table_q_loaded)
#plt.plot(cavity_length_simulation,table_q_loaded[:,0],'r-',label = 'thick mirror',marker = 'o')
plt.plot(cavity_length_simulation,table_q_loaded[:,1],'g-',label = 'thin mirror',marker = '*')
plt.plot(cavity_length_experiment,table_q_loaded[:,2],'b-',label = 'experiment',marker = '^')
plt.xlabel('Cavity Length(cm)')
plt.ylabel('Quality Factor(log)')
plt.legend()
plt.title('Quality Factors')
plt.autoscale(enable=True, axis='both', tight=None)
#plt.yscale('log')
plt.show()
#plt.savefig('Q.pdf', bbox_inches='tight')
#plt.close()
#table_q_unloaded = np.log(table_q_unloaded)

#plt.plot(cavity_length_simulation,resfre[:,0],'r-',label = 'thick mirror',marker = 'o')
# plt.plot(cavity_length_simulation,resfre[:,1],'g-',label = 'thin mirror',marker = '*',alpha = 0.8)
# #plt.plot(cavity_length_experiment,resfre[:,2],'b-',label = 'experiment',marker = '^',linewidth = 1)
# plt.plot(cavity_length_simulation,resfre[:,3],'y--',label = 'predicted',marker = 's')
# plt.plot(cavity_length_experiment,resfre[:,2],'b-',label = 'experiment',marker = '^')
# plt.errorbar(cavity_length_experiment,resfre[:,2],yerr = resfre[:,4])
# plt.xlabel('Cavity Length(cm)')
# plt.ylabel('Resonant Frequency(GHz)')
# plt.legend()
# plt.title('Resonant Frequencies')
# plt.tight_layout()
# plt.autoscale(enable=True, axis='both')
# plt.show()
#plt.savefig('Resonant_Frequencies.pdf', bbox_inches='tight')
#plt.close()


