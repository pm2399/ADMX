import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal 
from scipy.signal import find_peaks_cwt
import glob


def freqToPower(f,a,fo,q,c):
    return a*10**-6 * (1/((f-fo)**2+(fo/(2*q))**2)) + c

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

resfre_qual_4 = resfre_qual = np.ndarray(shape = (7,3))

cenpafolder = "Old_orpheus_simulations/"
filenames =[
	"HFSS_old_orpheus_s21_17cm.CSV",
	"HFSS_old_orpheus_s21_17p5cm.CSV",
	"HFSS_old_orpheus_s21_18cm.CSV",
	"HFSS_old_orpheus_s21_18p5cm.CSV",
	"HFSS_old_orpheus_s21_19cm.CSV",
	"HFSS_old_orpheus_s21_19p5cm.CSV",
	"HFSS_old_orpheus_s21_20p26cm.CSV"
	]

pzeros = [
	[0.000533,17.807,5000,0],
	[438,17.352,3000,0],
	[169.8,16.8741,3000,0],
	[43.4,16.422,3000,0],
	[16.05,15.9938,3000,0],
	[5.5256,15.5882,3000,0],
	[1.5012,15.0105,3000,0],
	]
for index in range(0,len(filenames)):
	#figure, graphs = plt.subplots(2, sharex=True)

	#open csv file
	A = np.genfromtxt(cenpafolder+filenames[index],delimiter = ',',skip_header=1)

	xlist = np.array(A[:,0])
	#i = np.where(xlist == (maxval-0.0015))
	#j = np.where(xlist == (maxval+0.0015))
	#print(i)
	#print(j)
	#print(A)
	xdata = A[:,0]
	#print(xdata)
	ydata = A[:,1]
	#linearize data
	ydata_lin= 10**(ydata/10)
	maxval = np.amax(ydata_lin)
	half_max = maxval/2
	half_max_near = find_nearest(ydata_lin,half_max)
	#print(maxval)
	#print(half_max_near)
	i = np.where(ydata_lin == maxval)
	firstnum = i[0]
	#print(firstnum)
	#spread = np.where(ydata_lin == half_max_near)
	#print(spread)
	if index == 0:
		initial = firstnum-310
		final =  firstnum+310
	elif index == 6:
		initial = firstnum-25
		final = firstnum+25
	else:
		initial = firstnum - 16
		final = firstnum + 16
	resonant_f= xdata[firstnum]
	decibel_f_1 = xdata[initial]
	decibel_f_2 = xdata[final]
	###f3d = xdata[half_max_near+8]-xdata[half_max_near]
	#print(f3d)
	spread = decibel_f_2 - decibel_f_1
	#print(initial)
	#print(final)
	data_extracted_x = np.array(xdata[initial[0]:final[0]])
	xlist = data_extracted_x[0:]
	data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
	ylist = data_extracted_y[0:]
	#print(xlist)
	#print(decibel_f_1)
	#print(decibel_f_2)
	#print(i)
	#print(resonant_f)
	#print(decibel_f)
	#quality = resonant_f/f3d
	#print(quality)
	plt.plot(xlist, ylist , 'b-',label='data')	##plot to the first graph
	#indexes = signal.find_peaks_cwt(ydata_lin, xdata)
	popt, pcov = curve_fit(freqToPower, xlist, ylist,p0=pzeros[index],maxfev=100000000)
	plt.plot(xlist, freqToPower(xlist,*popt),'r-',label='fit')
	resfre_qual = popt[1:3]
	#print(resfre_qual)
	coupling_coefficent = popt[1]/(1-(popt[1]))
	coupling_coefficent = np.absolute(coupling_coefficent)
	print("quality factor = " , popt[2])
	print("coupling_coefficent = ",coupling_coefficent)
	if coupling_coefficent>1.005:
		print("overcoupled")
	elif coupling_coefficent<0.995:
		print("undercoupled")
	else:
		print("critically coupled")
	print()
	#plt.show()
	plt.xlabel('frequency')
	plt.ylabel('power')
	#plt.legend()
	#plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	#plt.close()



