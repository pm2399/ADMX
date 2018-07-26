import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal 
from scipy.signal import find_peaks_cwt
import glob
import pandas as pd

def freqToPower(f,a,fo,q,c):
    return -a*10**-6 * (1/((f-fo)**2+(fo/(2*q))**2)) + c

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
font = {'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
resfre_qual_4 = resfre_qual = np.ndarray(shape = (7,3))

qual1 = np.ndarray(shape = (7,1))
q_unloaded = np.ndarray(shape = (7,1))
q_external = np.ndarray(shape = (7,1))
coupling_coefficent_vals = np.ndarray(shape = (7,1))
resonant_frequencies = np.ndarray(shape = (7,1))
background_level = np.ndarray(shape = (7,1))

cenpafolder = "thin_mirror_simulation_data/"
filenames =[
	"HFSS_old_orpheus_2mm_s11_17cm.CSV",
	"HFSS_old_orpheus_2mm_s11_17p5cm.CSV",
	"HFSS_old_orpheus_2mm_s11_18cm.CSV",
	"HFSS_old_orpheus_2mm_s11_18p5cm.CSV",
	"HFSS_old_orpheus_2mm_s11_19cm.CSV",
	"HFSS_old_orpheus_2mm_s11_19p5cm.CSV",
	"HFSS_old_orpheus_2mm_s11_21p26cm.CSV"
	]

pzeros = [
	[72.9,17.858,3000,0],
	[10.9371,17.3519,3000,0],
	[28.081,16.8741,3000,0],
	[72.0867,16.4221,3000,0],
	[69.3617,15.9939,3000,0],
	[66.4587,15.5878,3000,0],
	[62.034,15.0125,3000,0],
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
	minval = np.amin(ydata_lin)
	min 
	i = np.where(ydata_lin == minval)
	#print(i)
	firstnum = i[0]
	#print(firstnum)
	#spread = np.where(ydata_lin == half_max_near)
	#print(spread)
	#if index == 0:
		#initial = firstnum-40
		#final =  firstnum+310
	if index == 6:
		initial = firstnum-90
		final = firstnum+90
	else:
		initial = firstnum - 40
		final = firstnum + 40

	resonant_f= xdata[firstnum]
	decibel_f_1 = xdata[initial]
	decibel_f_2 = xdata[final]
	###f3d = xdata[half_max_near+8]-xdata[half_max_near]
	#print(f3d)
	#spread = decibel_f_2 - decibel_f_1
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
	plt.plot(xdata, ydata , 'b-',label='data')	##plot to the first graph
	#indexes = signal.find_peaks_cwt(ydata_lin, xdata)
	popt, pcov = curve_fit(freqToPower, xdata, ydata_lin,p0=pzeros[index],maxfev=100000000)
	plt.plot(xdata, 10*np.log10(freqToPower(xdata,*popt)),'r-',label='fit')
	#resfre_qual = popt[1:3]
	qual1[index] = popt[2]
	coupling_coefficent = minval/(1 - minval)
	coupling_coefficent = np.around(coupling_coefficent, decimals = 3)
	coupling_coefficent = np.absolute(coupling_coefficent)
	q_unloaded[index] = (1+coupling_coefficent)*qual1[index]
	q_external[index] = (coupling_coefficent*q_unloaded[index])
	coupling_coefficent_vals[index] = coupling_coefficent
	resonant_frequencies[index] = popt[1]
	background_level[index] = popt[3]
	#print(resfre_qual)
	# coupling_coefficent_fre = popt[1]/(1-(popt[1]))
	# #coupling_coefficent = 
	# coupling_coefficent = np.absolute(coupling_coefficent)
	# print("quality factor = " , popt[2])
	# print("coupling_coefficent = ",coupling_coefficent)
	# if coupling_coefficent>1.0005:
	# 	print("overcoupled")
	# elif coupling_coefficent<0.9995:
	# 	print("undercoupled")
	# else:
	# 	print("critically coupled")
	# print()
	#plt.show()
	popt = np.round(popt,decimals = 4)
	if index == 0:
		plt.text(17.75,-0.4,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.75,-0.45,"F = {}".format(popt[1]),fontdict = font)
	elif index == 1:
		plt.text(17.375,-0.3,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(17.375,-0.32,"F = {}".format(popt[1]),fontdict = font)
	elif index == 2:
		plt.text(16.9,-0.2,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.9,-0.22,"F = {}".format(popt[1]),fontdict = font)
	elif index == 3:
		plt.text(16.45,-0.12,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16.45,-0.13,"F = {}".format(popt[1]),fontdict = font)
	elif index == 4:
		plt.text(16,-0.06,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(16,-0.065,"F = {}".format(popt[1]),fontdict = font)
	elif index == 5:
		plt.text(15.65,-0.05,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(15.65,-0.053,"F = {}".format(popt[1]),fontdict = font)
	elif index == 6:
		plt.text(14.95,-0.03,"Q = {}".format(np.absolute(popt[2])),fontdict = font)
		plt.text(14.95,-0.032,"F = {}".format(popt[1]),fontdict = font)
	plt.xlabel('frequency')
	plt.ylabel('S11(db)')
	plt.legend()
	#plt.show()
	plt.savefig(filenames[index]+'.pdf', bbox_inches='tight')
	plt.close()

# Q_thin_mirror = pd.DataFrame(qual1)
# Q_thin_mirror.to_csv("Q_reflection_thin_mirror.CSV")
# # q_unloaded_thin_mirror = pd.DataFrame(q_unloaded)
# # q_unloaded_thin_mirror.to_csv("q_unloaded_reflection_thin_mirror.CSV")
# # q_external_thin_mirror = pd.DataFrame(q_external)
# # q_external_thin_mirror.to_csv("q_external_reflection_thin_mirror.CSV")
# #coupling_coefficent_thin_mirror = pd.DataFrame(coupling_coefficent_vals)
# #coupling_coefficent_thin_mirror.to_csv("coupling_coefficent_reflection_thin_mirror.CSV")
# resonant_frequencies_thin_mirror = pd.DataFrame(resonant_frequencies)
# resonant_frequencies_thin_mirror.to_csv("resonant_frequencies_reflection_thin_mirror.CSV")
# background_level_thin_mirror = pd.DataFrame(background_level)
# background_level_thin_mirror.to_csv("background_level_reflection_thin_mirror.CSV")

