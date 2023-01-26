from operator import le
import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

#print('Enter filepath:')
#path = input()
path = './postProcessing/forceCoeffs1/0/coefficient.dat'

def readPostProcess(path):
    with open(path) as fin:
        s = fin.readlines()
        s = s[13:]
        time, Cd, Cl = np.array([]), np.array([]), np.array([])
        for row in s:
            row = row.split('\t')
            time = np.append(time, float(row[0]))
            Cd = np.append(Cd, float(row[1]))
            Cl = np.append(Cl, float(row[3]))
    return time, Cd, Cl

time, Cd, Cl = readPostProcess(path)

plt.subplot(2, 1, 1)
plt.plot(time[5:], Cd[5:])
plt.xlabel('time')
plt.ylabel('Cd')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time[5:], Cl[5:])
plt.xlabel('time')
plt.ylabel('Cl')
plt.grid(True)
plt.tight_layout()
plt.savefig("CdCl.png")
#plt.show()
plt.clf()

#print('Enter start time (in seconds):')
#startTime = float(input())
startTime = float(5)

time = time[time > startTime]
Cd, Cl = Cd[-len(time):], Cl[-len(time):]

Cd_pulse = Cd - Cd.mean()
Cl_pulse = Cl - Cl.mean()

CdRMS, ClRMS = .0, .0
for val in Cd_pulse:
    CdRMS += val**2
for val in Cl_pulse:
    ClRMS += val**2
CdRMS = np.sqrt(CdRMS/len(Cd_pulse))
ClRMS = np.sqrt(ClRMS/len(Cl_pulse))
print(f'Mean Cd == {round(Cd.mean(), 5)}, mean Cl == {round(Cl.mean(), 5)}')
print(f'RMSE of Cd_pulse == {round(CdRMS, 5)}, RMSE of Cl_pulse == {round(ClRMS, 5)}')

plt.subplot(2, 1, 1)
plt.plot(time, Cd_pulse)
plt.xlabel('time')
plt.ylabel('Cd_pulse')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, Cl_pulse)
plt.xlabel('time')
plt.ylabel('Cl_pulse')
plt.grid(True)
plt.tight_layout()
plt.savefig("CdCl_pulse.png")
#plt.show()
plt.clf()

startInd, endInd = 0, 0
for i in range(1, len(Cl_pulse)-1):
    if Cl_pulse[i-1] < Cl_pulse[i] and Cl_pulse[i] > Cl_pulse[i+1]:
        startInd = i-1
        break
for i in range(len(Cl_pulse)-2, 1, -1):
    if Cl_pulse[i+1] < Cl_pulse[i] and Cl_pulse[i] > Cl_pulse[i-1]:
        endInd = i
        break

sp = rfft(Cl_pulse[startInd : endInd])
sp = np.abs(sp.real)/len(sp.real)
freq = rfftfreq(len(time[startInd : endInd]), d=time[1]-time[0])
max_amplitude_friq_index = np.argmax(np.abs(sp.real))
plt.bar(freq[:5*max_amplitude_friq_index], sp[:5*max_amplitude_friq_index], width=0.01)
plt.xlabel('frequency')
plt.ylabel('Cl_pulse_amplitude')
plt.savefig("amplitude-frequency.png")
#plt.show()
plt.clf()
print('fft of Cl_pulse')
print(f'max amplitude == {round(max(sp), 5)}, frequency of max amplitude == {round(freq[max_amplitude_friq_index], 5)}')
