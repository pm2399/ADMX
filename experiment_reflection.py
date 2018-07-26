import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import pandas as pd

def freqToPower(f,a,fo,q,c):
    return -a*10**-10 * (1/((f-fo)**2+(fo/(2*q))**2)) + c

resfre_qual = np.ndarray(shape = (7,3))

qual1 = np.ndarray(shape = (7,1))
q_unloaded = np.ndarray(shape = (7,1))
q_external = np.ndarray(shape = (7,1))
coupling_coefficent_vals = np.ndarray(shape = (7,1))
resonant_frequencies = np.ndarray(shape = (7,1))
background_level = np.ndarray(shape = (7,1))
font = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
cenpafolder = "Orpheus_empty_lcavitylength/"
filenames = [
	"ORPHEUS_EMPTY_17CM_S22.CSV",
	"ORPHEUS_EMPTY_17.5CM_S22.CSV",
	"ORPHEUS_EMPTY_18CM_S22.CSV",
	"ORPHEUS_EMPTY_18.5CM_S22.CSV",
	"ORPHEUS_EMPTY_19CM_S22.CSV",
	"ORPHEUS_EMPTY_19.5CM_S22.CSV",
	"ORPHEUS_EMPTY_20CM_S22.CSV",
]
pzeros = [
	[4000,17.956,3000,0],
	[7925,17.42,2000,0],
	[4392,16.956,3000,0],
	[3601,16.474,3000,0],
	[39478,16.075,2000,0],
	[12926,15.657,3000,0],
	[3234,15.275,3000,0],
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

	minval = np.amin(ydata_lin)
	i = np.where(ydata_lin == minval)
	#print(i)
	firstnum = i[0]
	#print(firstnum)
	initial = firstnum-40
	final = firstnum+40
	#print(initial)
	#print(final)
	# minval_data = np.amin(ydata)
	# index = np.where(ydata == minval)
	# #print(i)
	# firstnum_2 = index[0]
	# #print(firstnum)
	# initial_2 = firstnum_2-40
	# final_2 = firstnum_2+40

	data_extracted_x = np.array(xdata[initial[0]:final[0]])
	xlist = data_extracted_x[0:]
	data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
	ylist = data_extracted_y[0:]
	ylist_data = np.array(ydata[initial[0]:final[0]])
	# data_extracted_ydata = np.array(ydata[initial_2[0]:final_2[0]])
	# y_data_list = data_extracted_ydata[0:]
	##plot to the first graph
	#graphs[0].plot(xdata , ydata_lin)
	plt.plot(xlist , ylist_data, 'b-',label='data')
	#plt.plot(xdata, freqToPower(xdata, *pzeros[index]))

	##run the curve fit
	popt, pcov = curve_fit(freqToPower, xlist, ylist,p0=pzeros[index],maxfev=100000000)
	resfre_qual = popt[1:3]
	qual1[index] = popt[2]
	coupling_coefficent = minval/(1 - minval)
	coupling_coefficent = np.around(coupling_coefficent, decimals = 3)
	coupling_coefficent = np.absolute(coupling_coefficent)
	q_unloaded[index] = (1+coupling_coefficent)*qual1[index]
	q_external[index] = (coupling_coefficent*q_unloaded[index])
	coupling_coefficent_vals[index] = coupling_coefficent
	resonant_frequencies[index] = popt[1]
	background_level[index] = popt[3]
	print(resfre_qual)
	print("quality factor = " , popt[2])
	print("coupling_coefficent = ",coupling_coefficent)
	if coupling_coefficent>1.005:
	 	print("overcoupled")
	elif coupling_coefficent<0.995:
	 	print("undercoupled")
	else:
	 	print("critically coupled")
	print()
	##plot to the second graph
	#graphs[1].plot(xdata, freqToPower(xdata,*popt))
	plt.plot(xlist, 10*np.log10(freqToPower(xlist,*popt)),'r-',label='fit')
	#print(popt)
	#plt.show()
	plt.xlabel('frequency')
	plt.ylabel('S11(dB)')
	plt.legend()
	str_q = "Q = {}".format(np.absolute(popt[2]))
	str_f = "F = {}".format(popt[1])
	#coupling_coefficent = np.around(coupling_coefficent,decimals = 4)
	coupling_coefficent = np.format_float_scientific(coupling_coefficent, unique=False, precision=5)
	popt = np.round(popt,decimals = 4)
	if index == 0:
		plt.text(17.85,-15,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.85,-16,"F = {}".format(popt[1]),fontdict = font)
		#plt.text(17.85,-15,str_q\nstr_f,fontdict = font)
	elif index == 1:
		plt.text(17.475,-9,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.475,-9.3,"F = {}".format(popt[1]),fontdict = font)
	elif index == 2:
		plt.text(17,-10,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17,-10.5,"F = {}".format(popt[1]),fontdict = font)
	elif index == 3:
		plt.text(16.55,-10,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.55,-10.5,"F = {}".format(popt[1]),fontdict = font)
	elif index == 4:
		plt.text(16.1,-10,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.1,-10.3,"F = {}".format(popt[1]),fontdict = font)
	elif index == 5:
		plt.text(15.7,-12,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.7,-12.5,"F = {}".format(popt[1]),fontdict = font)
	elif index == 6:
		plt.text(15.325,-10,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.325,-10.5,"F = {}".format(popt[1]),fontdict = font)
	#plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	#plt.close()
	##show total plot
	plt.show()
    

# Q_thin_mirror = pd.DataFrame(qual1)
# Q_thin_mirror.to_csv("Q_reflection_experiment.CSV")
# # q_unloaded_thin_mirror = pd.DataFrame(q_unloaded)
# # q_unloaded_thin_mirror.to_csv("q_unloaded_reflection_thin_mirror.CSV")
# # q_external_thin_mirror = pd.DataFrame(q_external)
# # q_external_thin_mirror.to_csv("q_external_reflection_thin_mirror.CSV")
# #coupling_coefficent_thin_mirror = pd.DataFrame(coupling_coefficent_vals)
# #coupling_coefficent_thin_mirror.to_csv("coupling_coefficent_reflection_thin_mirror.CSV")
# resonant_frequencies_thin_mirror = pd.DataFrame(resonant_frequencies)
# resonant_frequencies_thin_mirror.to_csv("resonant_frequencies_reflection_experiment.CSV")
# background_level_thin_mirror = pd.DataFrame(background_level)
# background_level_thin_mirror.to_csv("background_level_reflection_experiment.CSV")





