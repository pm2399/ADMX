import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd

def freqToPower(f,a,fo,q,c):
    return a*10**-10 * (1/((f-fo)**2+(fo/(2*q))**2)) + c

qual3 = np.ndarray(shape = (7,1))
q_unloaded = np.ndarray(shape = (7,1))
q_external = np.ndarray(shape = (7,1))
coupling_coefficent_vals = np.ndarray(shape = (7,1))
resonant_frequencies = np.ndarray(shape = (7,1))
background_level = np.ndarray(shape = (7,1))
maintext = np.array(['Q = ','g = ','F = '])
font = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

cenpafolder="Orpheus_empty_lcavitylength/"
filenames = [
	"ORPHEUS_EMPTY_17CM_S21.CSV",
	"ORPHEUS_EMPTY_17.5CM_S21.CSV",
	"ORPHEUS_EMPTY_18CM_S21.CSV",
	"ORPHEUS_EMPTY_18.5CM_S21.CSV",
	"ORPHEUS_EMPTY_19CM_S21.CSV",
	"ORPHEUS_EMPTY_19.5CM_S21.CSV",
	"ORPHEUS_EMPTY_20CM_S21.CSV",
]
pzeros = [
	[0.506,17.956,3000,0],
	[0.523,17.42,3000,0],
	[0.359,16.956,2000,0],
	[0.550,16.474,2000,0],
	[0.581,16.075,2000,0],
	[0.215,15.657,2000,0],
	[0.175,15.275,2000,0],
	]
for index in range(0,len(filenames)):
	#figure, graphs = plt.subplots(2, sharex=True)

	#open csv file
	A = np.genfromtxt(cenpafolder+filenames[index],delimiter = ',',skip_header=1,)
	#print(A)
	#parse data
	xdata = A[:,0]/(10**9)
	ydata = A[:,1]
	#linearize data
	ydata_lin= 10**(ydata/10)
	if index == 0 :
		xdata = xdata[50:]
		ydata_lin = ydata_lin[50:]
		ydata = ydata[50:]

	maxval = np.amax(ydata_lin)
	max_s21 = np.amax(ydata)
	i = np.where(ydata_lin == maxval)
	firstnum = i[0]
	initial = firstnum-20
	final = firstnum+20
	data_extracted_x = np.array(xdata[initial[0]:final[0]])
	xlist = data_extracted_x[0:]
	data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
	ylist = data_extracted_y[0:]
	ylist_data = np.array(ydata[initial[0]:final[0]])
	##plot to the first graph
	#graphs[0].plot(xdata , ydata_lin)
	plt.plot(xdata, ydata , 'b-',label='data')
	  #plt.plot(xdata, freqToPower(xdata, *pzeros[index]))

	##run the curve fit
	popt, pcov = curve_fit(freqToPower, xdata,ydata_lin,p0=pzeros[index],maxfev=100000000)
	#popt = np.round(popt,decimals = 4)
	#print(popt)
	##plot to the second graph
	#graphs[1].plot(xdata, freqToPower(xdata,*popt))
	plt.plot(xdata,10*np.log10(freqToPower(xdata,*popt)),'r-',label='fit')
	qual3[index] = popt[2]
	coupling_coefficent = maxval/(1 - maxval)
	#coupling_coefficent = np.around(coupling_coefficent, decimals = 3)
	coupling_coefficent = np.absolute(coupling_coefficent)
	q_unloaded[index] = (1+coupling_coefficent)*qual3[index]
	q_external[index] = (coupling_coefficent*q_unloaded[index])
	coupling_coefficent_vals[index] = coupling_coefficent
	resonant_frequencies[index] = popt[1]
	background_level[index] = popt[3]
	# print("quality factor = " , popt[2])
	# print("coupling_coefficent = ",coupling_coefficent)
	# if coupling_coefficent>1.01:
	# 	print("overcoupled")
	# elif coupling_coefficent<0.99:
	# 	print("undercoupled")
	# else:
	# 	print("critically coupled")
	# print()
	plt.xlabel('frequency')
	plt.ylabel('S21(dB)')
	plt.legend()
	#popt = np.round(popt,decimals = 4)
	#coupling_coefficent = np.around(coupling_coefficent,decimals = 4)
	coupling_coefficent = np.format_float_scientific(coupling_coefficent, unique=False, precision=5)
	popt = np.round(popt,decimals = 4)
	if index == 0:
		plt.text(17.7,-55,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.7,-58,"F = {}".format(popt[1]),fontdict = font)
		plt.text(17.7,-61,"g = {}".format(coupling_coefficent),fontdict = font)
		#plt.text(17.85,-15,str_q\nstr_f,fontdict = font)
	elif index == 1:
		plt.text(17,-55,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17,-58,"F = {}".format(popt[1]),fontdict = font)
		plt.text(17,-61,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 2:
		plt.text(16.5,-55,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.5,-58,"F = {}".format(popt[1]),fontdict = font)
		plt.text(16.5,-61,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 3:
		plt.text(16,-55,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16,-58,"F = {}".format(popt[1]),fontdict = font)
		plt.text(16,-61,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 4:
		plt.text(15.5,-55,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.5,-58,"F = {}".format(popt[1]),fontdict = font)
		plt.text(15.5,-61,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 5:
		plt.text(15,-60,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15,-63,"F = {}".format(popt[1]),fontdict = font)
		plt.text(15,-66,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 6:
		plt.text(15.4,-60,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.4,-63 ,"F = {}".format(popt[1]),fontdict = font)
		plt.text(15.4,-66,"g = {}".format(coupling_coefficent),fontdict = font)
	# plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	# plt.close()
	##show total plot
	plt.show()
# print(qual3)
# Q_experiment = pd.DataFrame(qual3)
# Q_experiment.to_csv("Q_experiment.CSV")
# q_unloaded_experiment = pd.DataFrame(q_unloaded)
# q_unloaded_experiment.to_csv("q_unloaded_experiment.CSV")
# q_external_experiment = pd.DataFrame(q_external)
# q_external_experiment.to_csv("q_external_experiment.CSV")
# coupling_coefficent_experiment = pd.DataFrame(coupling_coefficent_vals)
# coupling_coefficent_experiment.to_csv("coupling_coefficent_experiment.CSV")
# resonant_frequencies_experiment = pd.DataFrame(resonant_frequencies)
# resonant_frequencies_experiment.to_csv("resonant_frequencies_experiment.CSV")
# background_level_experiment = pd.DataFrame(background_level)
# background_level_experiment.to_csv("background_level_experiment.CSV")
