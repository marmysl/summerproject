""" Impact Analysis 
Maria Mysliwiec """

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import peakutils
from peakutils.plot import plot as pplot
from scipy.signal import detrend
from matplotlib import pyplot as plt
import os

# FUNCTIONS
def stats(i): #return stats
   
    #calculate standard deviation, mean, rms, subtracted
    sigma=np.std(i)
    avg=np.mean(i)
    subt=i-avg
    rms=np.sqrt(np.mean(subt**2))
    
    return sigma,avg,subt,rms

# find factors for dimensions of bin when calculating area
def print_factors(x):   
    print("The factors of",x,"are:")
    for i in range(1, x + 1):
        if x % i == 0:
            print(i)

#change directory so that it corresponds to where impact/background files are
os.chdir("/Users/mariamysliwiec/PythonProj") 

# Enter impact filename and check if it exists
while True:
    filename= input("Enter the impact data file name:")
    if os.path.exists(filename) == True:
        break
        
#Format columns
df = pd.read_csv(filename,header=None, usecols=[1,2,4],names=['Time','Current','Voltage'] )

s=df['Time']
i=df['Current']
v=df['Voltage']

# Enter background filename and check if it exists
while True:
    filename2= input("Enter the background data file name:")
    if os.path.exists(filename2) == True:
        break

#Format columns
df2 = pd.read_csv(filename2,header=None, usecols=[1,2,4],names=['Time','Current','Voltage'] )

snoise=df2['Time']
inoise=df2['Current']
vnoise=df2['Voltage']

#Absolute value of current values
i=abs(i)

#Detrend data; set baseline to zero by subtracting mean from each point (constant type)
    # Set type as linear to subtract linear least-squares fit from data.
itrnd=detrend(i,type='constant')
inoisetrnd=detrend(inoise,type='constant')

# absolute value of detrended data
itrnd=abs(itrnd)
inoisetrnd=abs(inoisetrnd)

#return stats
sigmai,meani,subcurrent,rms_signal=stats(itrnd)
sigmanoise,meannoise,backsub,rmsnoise=stats(inoisetrnd)

#plot baseline cut data
plt.plot(s,itrnd,label='impact2')
plt.plot(snoise,inoisetrnd,label='noise2')
plt.title('Impact vs Noise')
plt.legend()
#plt.xlim(0,10)
plt.show()

#Smooth data
isubt=itrnd-inoisetrnd #subtracting noise from i detrended

plt.plot(s,itrnd,label='impact2')
plt.plot(snoise,inoisetrnd,label='noise2')
plt.plot(s,isubt,label='subtracted',color='red')
plt.title('Smoothed Data')
plt.legend()
#plt.xlim(9.7,9.73)
plt.show()

#Savitzky Golay Filter: 
from scipy.signal import savgol_filter
ihat = savgol_filter(itrnd, 27, 3) # window size, polynomial

plt.plot(s,itrnd,label='impact')
plt.plot(s,ihat, color='red',label='filtered')
plt.title('Savitzky Golay Filter')
#plt.xlim(9.7,9.73)
plt.show()

#Peak detection
indexes = peakutils.indexes(ihat,thres=5*sigmanoise,min_dist=100,thres_abs=True) 
#set threshold as an ABSOLUTE min

plt.figure(figsize=(10,6))
pplot(s, ihat, indexes)
plt.title('Peak Identification')
#plt.xlim(7.53,7.56)
plt.show()

#Find Width
# The width is any continuous section of data around the peak that is above the noise floor 
peakwidths = []
for x in ihat: 
        if sigmanoise < x:
            peakwidths.append(x)
            
widthindices = np.where(np.in1d(ihat, peakwidths))[0]  #return indexes (for peakwidths in ihat)
timeindices= s[widthindices]

plt.figure(figsize=(12,8))
plt.plot(s[widthindices],peakwidths, 'go')
pplot(s,ihat,indexes)
#plt.xlim(2,2.5)

#Plot graph of only peaks
peakwidths=detrend(peakwidths,type='constant')

plt.figure(figsize=(12,8))
plt.plot(s[widthindices],peakwidths, 'g')
plt.title('Data above Noise Floor')
#plt.xlim(9,10)

# AREA: FIND CUMULATIVE SUM

peakwidths=np.array(peakwidths)
secs=np.array(s[widthindices])

areascu = np.cumsum(peakwidths)

plt.figure()
plt.plot(secs,areascu)
plt.plot(secs,peakwidths, 'g')
plt.title('Cumulative Sum')

#Absolute value of Cumulative Sum
posareacu=abs(areascu)
plt.figure()
plt.plot(secs,posareacu)
plt.title('Absolute Value of Cumulative Sum')

#Find factors for 2D array
print_factors(len(peakwidths))

# AREA CALCULATION USING HISTOGRAM
# set up some parameters for the plot

bins = 551 #ensure that bin value is divisible by the length of peakwidths variable, use factors function to manually input value
index = np.arange(bins)
bar_width = 1
opacity = 0.4

#conversion of a numpy array from 1D to 2D
peaksbin = np.reshape(peakwidths,(bins,2)) #columns value is the number of points in each bin

#Compute the areas for each part using the composite trapezoidal rule.
areas1 = np.trapz(peaksbin,dx=0.001)

# plot the areas
fig, (ax1) = plt.subplots(nrows=1,sharex=True,sharey=True,figsize = (8, 8))
ax1.bar(index, areas1, bar_width, alpha=opacity, color='black')

ax1.tick_params(axis='both',which='both',bottom='off',top='off',right='off')

fig.text(0.04, 0.5, 'AUC', ha='center',fontsize=18,rotation='vertical')
ax1.set_title('Area under the curve (AUC)', fontsize=20)
plt.show()

#Define positive area and append into its own array
posarea = []

for x in areas1: 
    if x > 0:
        posarea.append(x)
        
#Export Data Function
import csv

#single columns of data
with open('areacalculation.asc', 'w+') as f:
    writer = csv.writer(f)
    for x in areas1:
        writer.writerow([x])

#repeat process per array of data you want to export into an asc file, 
#make sure the filename is different each time so the data is not overwritten
        