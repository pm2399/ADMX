import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal 
from scipy.signal import find_peaks_cwt
import glob
import pandas as pd


def freqToPower(f,a,fo,q,c):
    return a*10**-6 * (1/((f-fo)**2+(fo/(2*q))**2)) + c

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
cenpafolder = "thin_mirror_simulation_data/"
filenames =[
	"HFSS_old_orpheus_2mm_s21_17cm.CSV",
	"HFSS_old_orpheus_2mm_s21_17p5cm.CSV",
	"HFSS_old_orpheus_2mm_s21_18cm.CSV",
	"HFSS_old_orpheus_2mm_s21_18p5cm.CSV",
	"HFSS_old_orpheus_2mm_s21_19cm.CSV",
	"HFSS_old_orpheus_2mm_s21_19p5cm.CSV",
	"HFSS_old_orpheus_2mm_s21_21p26cm.CSV"
	]

pzeros = [
	[44.5,17.858,3000,0],
	[10.9371,17.3519,3000,0],
	[2.8081,16.8741,3000,0],
	[0.61197,16.4221,3000,0],
	[0.159,15.9939,3000,0],
	[0.054657,15.5878,3000,0],
	[0.02092,15.0125,3000,0],
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
	max_s21 = np.amax(ydata)
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
		initial = firstnum-40
		final =  firstnum+310
	if index == 6:
		initial = firstnum-90
		final = firstnum+90
	else:
		initial = firstnum - 40
		final = firstnum + 40

	resonant_f= xdata[firstnum]
	decibel_f_1 = xdata[initial]
	decibel_f_2 = xdata[final]
	##f3d = xdata[half_max_near+8]-xdata[half_max_near]
	# print(f3d)
	# spread = decibel_f_2 - decibel_f_1
	# print(initial)
	# print(final)
	data_extracted_x = np.array(xdata[initial[0]:final[0]])
	xlist = data_extracted_x[0:]
	data_extracted_y = np.array(ydata_lin[initial[0]:final[0]])
	ylist = data_extracted_y[0:]
	ydata_list = np.array(ydata[initial[0]:final[0]])
	#print(xlist)
	#print(decibel_f_1)
	#print(decibel_f_2)
	#print(i)
	#print(resonant_f)
	#print(decibel_f)
	#quality = resonant_f/f3d
	#print(quality)
	#plt.plot(xdata, ydata , 'b-',label='data')	##plot to the first graph
	#indexes = signal.find_peaks_cwt(ydata_lin, xdata)
	plt.plot(xlist, ydata_list , 'b-',label='data')
	popt, pcov = curve_fit(freqToPower, xlist, ylist,p0=pzeros[index],maxfev=100000000)
	plt.plot(xlist, 10*np.log10(freqToPower(xlist,*popt)),'r-',label='fit')
	#plt.plot(xdata, ydata , 'b-',label='data')

	qual1[index] = popt[2]
	#print(resfre_qual)
	coupling_coefficent = maxval/(1 - maxval)
	#coupling_coefficent = np.around(coupling_coefficent, decimals = 3)
	coupling_coefficent = np.absolute(coupling_coefficent)
	q_unloaded[index] = (1+coupling_coefficent)*qual1[index]
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
	#plt.show()
	plt.xlabel('frequency')
	plt.ylabel('S21(dB)')
	plt.legend()
	#plt.show()
	print(popt)
	coupling_coefficent = np.format_float_scientific(coupling_coefficent, unique=False, precision=5)
	popt = np.round(popt,decimals = 4)
	#print(popt)
	if index == 0:
		plt.text(17.85,-35,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.85,-38,"F = {}".format(popt[1]),fontdict = font)
		plt.text(17.85,-41,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 1:
		plt.text(17.344,-40,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.344,-42,"F = {}".format(popt[1]),fontdict = font)
		plt.text(17.344,-44,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 2:
		plt.text(16.866,-45,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.866,-47,"F = {}".format(popt[1]),fontdict = font)
		plt.text(16.866,-49,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 3:
		plt.text(16.414,-48,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.414,-50,"F = {}".format(popt[1]),fontdict = font)
		plt.text(16.414,-52,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 4:
		plt.text(15.99,-52,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.99,-53,"F = {}".format(popt[1]),fontdict = font)
		plt.text(15.99,-54,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 5:
		plt.text(15.573,-60,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.573,-62,"F = {}".format(popt[1]),fontdict = font)
		plt.text(15.573,-64,"g = {}".format(coupling_coefficent),fontdict = font)
	elif index == 6:
		plt.text(14.995,-62.5,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(14.995,-64,"F = {}".format(popt[1]),fontdict = font)
		plt.text(14.995,-65.5,"g = {}".format(coupling_coefficent),fontdict = font)
	plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	plt.close()
	#plt.show()
# print(qual1)
# Q_thin_mirror = pd.DataFrame(qual1)
# Q_thin_mirror.to_csv("Q_thin_mirror.CSV")
# q_unloaded_thin_mirror = pd.DataFrame(q_unloaded)
# q_unloaded_thin_mirror.to_csv("q_unloaded_thin_mirror.CSV")
# q_external_thin_mirror = pd.DataFrame(q_external)
# q_external_thin_mirror.to_csv("q_external_thin_mirror.CSV")
# coupling_coefficent_thin_mirror = pd.DataFrame(coupling_coefficent_vals)
# coupling_coefficent_thin_mirror.to_csv("coupling_coefficent_thin_mirror.CSV")
# #resonant_frequencies_thin_mirror = pd.DataFrame(resonant_frequencies)
# #resonant_frequencies_thin_mirror.to_csv("resonant_frequencies_thin_mirror.CSV")
# #background_level_thin_mirror = pd.DataFrame(background_level)
# #background_level_thin_mirror.to_csv("background_level_thin_mirror.CSV")








