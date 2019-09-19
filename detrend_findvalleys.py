#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:52:35 2019

@author: mariamysliwiec
"""

#Pop-up: Importing CSV Data
#import tkinter as tk
#from tkinter import filedialog
#import pandas as pd

#root= tk.Tk()

#canvas1 = tk.Canvas(root, width = 300, height = 300, bg = 'lightsteelblue2', relief = 'raised')
#canvas1.pack()

#def getCSV ():
#    global df
    
#    import_file_path = filedialog.askopenfilename()
#    df = pd.read_csv (import_file_path)
#    print (df)
    
#browseButton_CSV = tk.Button(text="Import CSV File", command=getCSV,bg='green',fg='white',font=('helvetica', 12, 'bold'))
#canvas1.create_window(150, 150, window=browseButton_CSV)

#root.mainloop()

#DETRENDING DATA
#import numpy as np
#t = np.linspace(0,5,100)
#
#print(t)
#
#x=t+np.random.normal(size=100)
#
#print(x)
#
#from scipy import signal
#xdetrended=signal.detrend(x)
#
#import matplotlib.pyplot as plt
#plt.figure(figsize=(5,4))
#plt.plot(t,x,label='x')
#plt.plot(t,xdetrended,label='Detrended')
#plt.legend()
#
#plt.show()
#
#import pandas as pd
#import numpy as np
#from scipy import signal
#import matplotlib.pyplot as plt
#import peakutils
#
#def read():
#	
#	filename= input("Enter the file name:")  
#	df = pd.read_csv(filename,header=None, usecols=[1,2,4],names=['Time','Current','Voltage'] )
#    
#    #Assignments
#	s=df['Time']
#	i=df['Current']
#	v=df['Voltage']	
#	return s,i,v
#
#def stats(i):
#	
#	#Calculate the standard deviation, mean, 'backsub', and rms
#	sigma_noise=np.std(i)
#	meannoise=np.mean(i)
#	backsub=i-meannoise   
#	rmsnoise=np.sqrt(np.mean(backsub**2))
#    
#	return sigma_noise,meannoise,backsub,rmsnoise
#
#def main(args):
#    
#    time,current,potential=read()
#    sigma,mean,subcurrent,rms_signal=stats(current)
#    subcurrent=(subcurrent)/1e-12
#    
#    plt.plot(time,subcurrent,label="Normal")
#    
#    sigma_noise=2.3
#    
#    subcurrent_detrended=signal.detrend(subcurrent,axis=0,type='linear')
#    
#    indexes=peakutils.indexes(subcurrent_detrended,thres=4*sigma_noise)
#    
#    plt.plot(time,subcurrent_detrended,label="Detrended")
#    plt.legend() 
#    plt.show()
#    
#    return 0
#		
#if __name__ == '__main__':
#    import sys
#    sys.exit(main(sys.argv))

#All the libraries for specific functions
import numpy as np
import peakutils
from peakutils.plot import plot as pplot
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal

import matplotlib
matplotlib.use('Agg') 

#Impact file upload
filename= input("Enter the impact data file name:")  
df = pd.read_csv(filename,header=None, usecols=[1,2,4],names=['Time','Current','Voltage'] )
    
#Assignments for Impact
s=df['Time']
i=df['Current']
v=df['Voltage']	

#Noise file upload
filename2= input("Enter the background noise file name:")  
df = pd.read_csv(filename2,header=None, usecols=[1,2,4],names=['Time','Current','Voltage'] )
    
#Assignments for Noise
snoise=df['Time']
inoise=df['Current']
vnoise=df['Voltage']	

#Plot comparing original noise vs. impact data alignment
plt.plot(s,i,label='impact1')
plt.plot(snoise,inoise,label='noise1')
plt.legend()
#plt.xlim(9.7,9.81) #set x limit so that they are the same range (in this example only)
plt.show()

#Detrend both the noise and the impact data in order to ensure they are aligned on same axis
betterinoise=signal.detrend(inoise,axis=0,type='linear')
betteri=signal.detrend(i,axis=0,type='linear')

#Plot to compare
plt.plot(s,betteri,label='impact2')
plt.plot(snoise,betterinoise,label='noise2')
plt.legend()
#plt.xlim(9.7,9.81)
plt.show()

# What is the standard deviation of the noise
stdnoise=np.std(betterinoise)

#betteri_rect = np.absolute(betteri)
#plt.plot(s,betteri_rect)
#plt.show()

smoothi=signal.hilbert(betteri)
plt.figure(figsize=(10,6))  #zoom in
plt.plot(s, smoothi,label='smoothened')

#Find the peaks using the peakutils library functions
indexes = peakutils.indexes(betteri,thres=6*stdnoise,min_dist=100,thres_abs=True) #set 4*STD as an ABSOLUTE min

# Find peaks(max).
peak_indexes = signal.argrelextrema(betteri, np.greater)
peak_indexes = peak_indexes[0]
 
# Find valleys(min).
valley_indexes = signal.argrelextrema(betteri, np.less)
valley_indexes = valley_indexes[0]

#the plot with detected peaks
plt.figure(figsize=(10,6))  #zoom in
pplot(s, betteri, indexes)
plt.title('Peak Identification')

(fig, ax) = plt.subplots()
ax.plot(s, betteri)
 
# Plot peaks.
peak_x = peak_indexes
peak_y = betteri[peak_indexes]
ax.plot(peak_x, peak_y, marker='o', linestyle='dashed', color='green', label='Peaks')
 
# Plot valleys.
valley_x = valley_indexes
valley_y = betteri[valley_indexes]
ax.plot(valley_x, valley_y, marker='o', linestyle='dashed', color='red', label='Valleys')
