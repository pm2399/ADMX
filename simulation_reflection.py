import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob


def freqToPower(f,a,fo,q,c):
    return -a*10**-10 * (1/((f-fo)**2+(fo/(2*q))**2)) + c

resfre_qual_1 = np.ndarray(shape = (7,3))

cenpafolder="Old_orpheus_simulations/"
filenames = [
	"HFSS_old_orpheus_s11_17cm.CSV",
	"HFSS_old_orpheus_s11_17p5cm.CSV",
	"HFSS_old_orpheus_s11_18cm.CSV",
	"HFSS_old_orpheus_s11_18p5cm.CSV",
	"HFSS_old_orpheus_s11_19cm.CSV",
	"HFSS_old_orpheus_s11_19p5cm.CSV",
	"HFSS_old_orpheus_s11_20p26cm.CSV"
]
pzeros = [
	[3098,17.807,5000,0],
	[4320,17.352,30000,0],
	[5883,16.8741,3000,0],
	[6392,16.422,3000,0],
	[6432,15.9938,3000,0],
	[6369,15.5882,3000,0],
	[6050,15.0105,3000,0],
	]
for index in range(0,len(filenames)):
	#figure, graphs = plt.subplots(2, sharex=True)

	#open csv file
	A = np.genfromtxt(cenpafolder+filenames[index],delimiter = ',',skip_header=1,)
	#print(A)
	#parse data
	xdata = A[:,0]
	ydata = A[:,1]
	#linearize data
	ydata_lin= 10**(ydata/10)
	#if index == 0 :
		#xdata = xdata[50:]
		#ydata_lin = ydata_lin[50:]


	minval = np.amin(ydata_lin)
	i = np.where(ydata_lin == minval)
	#print(i)
	firstnum = i[0]
	#print(firstnum)
	if index == 0:
		initial = firstnum-440
		final = firstnum+440
	else:
		initial = firstnum-60
		final = firstnum+60
	#print(initial)
	#print(final)
	data_extracted_x = np.array(xdata[initial[0]:final[0]])
	xlist = data_extracted_x[0:]
	data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
	ylist = data_extracted_y[0:]

	##plot to the first graph
	#graphs[0].plot(xdata , ydata_lin)
	plt.plot(xlist , ylist, 'b-',label='data')
	#plt.plot(xdata, freqToPower(xdata, *pzeros[index]))

	##run the curve fit
	popt, pcov = curve_fit(freqToPower, xlist, ylist,p0=pzeros[index],maxfev=100000000)
	resfre_qual = popt[1:3]
	#print(resfre_qual)
	coupling_coefficent = popt[2]/(1-(popt[2]))
	coupling_coefficent = np.absolute(coupling_coefficent)
	# print(coupling_coefficent)
	# if coupling_coefficent>1:
	# 	print("overcoupled")
	# elif coupling_coefficent ==1:
	# 	print("critically coupled")
	# else:
	# 	print("undercoupled")
	##plot to the second graph
	#graphs[1].plot(xdata, freqToPower(xdata,*popt))
	plt.plot(xlist, freqToPower(xlist,*popt),'r-',label='fit')
	#print(popt)
	#plt.show()
	plt.xlabel('frequency')
	plt.ylabel('power')
	plt.legend()
	plt.show()
	#plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	#plt.close()
	##show total plot
	#plt.show()







