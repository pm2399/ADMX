import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.offsetbox import AnchoredText

def freqToPower(f,a,fo,q,c):
	return a*10**-10 * (1/((f-fo)**2+(fo/(2*q))**2))+c*10**(-7)

cepafolder = "tuner_data/"
filenames = [
	"ORPHEUS_S21_N16AVG_1.CSV",
	"ORPHEUS_S21_N16AVG_2.CSV"
]

pzeros = [
	[0.031207,15.245,3000,0],
	[0.0272359,15.24,3000,0]
	]

font = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
for index in range(0,len(filenames)):
	A = np.genfromtxt(cepafolder + filenames[index], delimiter = ',')
	xdata = A[:,0]/(10**9)
	ydata = A[:,1]
	#linearize data
	ydata_lin= 10**(ydata/10)
	#plt.plot(xdata,ydata_lin)
	#plt.show()
	maxval = np.amax(ydata_lin)
	max_s21 = np.amax(ydata)
	i = np.where(ydata_lin == maxval)
	firstnum = i[0]
	initial = firstnum-50
	final = firstnum+50
	data_extracted_x = np.array(xdata[initial[0]:final[0]])
	xlist = data_extracted_x[0:]
	data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
	ylist = data_extracted_y[0:]
	ylist_data = np.array(ydata[initial[0]:final[0]])
	##plot to the first graph
	#graphs[0].plot(xdata , ydata_lin)
	plt.plot(xdata , ydata , 'b-',label='data')
	  #plt.plot(xdata, freqToPower(xdata, *pzeros[index]))

	##run the curve fit
	popt, pcov = curve_fit(freqToPower, xdata, ydata_lin,p0=pzeros[index],maxfev=100000000)
	popt = np.round(popt,decimals = 4)
	print(popt)
	##plot to the second graph
	#graphs[1].plot(xdata, freqToPower(xdata,*popt))
	plt.plot(xdata, 10*np.log10(freqToPower(xdata,*popt)),'r-',label='fit')
	#print(freqToPower(xlist,*popt))
	#plt.plot(xlist,10*np.log10(freqToPower(xlist,*popt)))
	#print(popt)
	#popt = np.round(popt,decimals = 4)
	#print(popt)
	#print(freqToPower(xlist,*popt))
	#plt.plot(xlist,10*np.log10(freqToPower(xlist,*popt)))
	#plt.show()

	if index == 0:
		plt.text(15.4,-55,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.4,-58,"F = {}".format(popt[1]),fontdict = font)
	elif index == 1:
		plt.text(15.4,-60,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.4,-63,"F = {}".format(popt[1]),fontdict = font)
	# qual[index] = popt[2]
	# coupling_coefficent = maxval/(1 - maxval)
	# #coupling_coefficent = np.around(coupling_coefficent, decimals = 3)
	# coupling_coefficent = np.absolute(coupling_coefficent)
	# q_unloaded[index] = (1+coupling_coefficent)*qual[index]
	# q_external[index] = (coupling_coefficent*q_unloaded[index])
	# coupling_coefficent_vals[index] = coupling_coefficent
	# resonant_frequencies[index] = popt[1]
	# background_level[index] = popt[3]
	# print("quality factor = " , popt[2])
	# print("coupling_coefficent = ",coupling_coefficent)
	# if coupling_coefficent>1.01:
	# 	print("overcoupled")
	# elif coupling_coefficent<0.99:
	#  	print("undercoupled")
	# else:
	#  	print("critically coupled")
	# print()
	plt.xlabel('frequency')
	plt.ylabel('S21(dB)')
	plt.legend()
	plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	plt.close()
	#plt.show()
