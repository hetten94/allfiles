#! /usr/bin python3.5
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

tauc = 4*10**(-12)#с - время захвата электрона
taur = 100*10**(-12)#с - рекомбинационное время жизни
taus = 170*10**(-15)#с - время рассеяния
r = 10**(-3);# см размер пучка
R0 = 0.2;# см радиус кривизны линзы
nSi = 3.48;# показатель преломления кремния
w0 = 10**(-3);# см размер пучка в перетяжке
nair = 1;# показатель преломления воздуха
nInGaAs = 3.5;# показатель преломления InGaAs
I0 = 2.9*10**(17);# Эрг/(см^2*c)
v = 2*10**14;# Гц частота лазера
taurate = 100*10**(-15);# с время длительности импульса
T = 10**(-4);#см активный слой подложки
W = 2*10**(-3);# см ширина электрода
nue = 1.2*10**(5);# см^2/(ед.СГС*с) подвижность электронов
nu = 1000# фактор экранирования
alpha = 1.04*10**(4);# см^(-1) поглощение
e = 4.8*10**(-10);# ед. заряда СГС
c = 3*10**(10);# см/с скорость света
L = 2*10**(-3);# cм зазор
A = W*L;# площадь зазора
z = 1;# расстояние от генератора до детектора
Eb = 50;# ед.СГС - напряженность поля смещения
N = 2001;# кол-во точек
T0 = 12.5*10**(-12);# c временной интервал
R = ((nInGaAs-nair)/(nInGaAs+nair))**2;
h = 6.6*10**(-27);# Эрг*с
W0 = I0*alpha*(1-R)/(h*v)*math.exp(-2*r*r/w0/w0);
me = 0.2457*10**(-28)# г - эффективная масса электрона в InGaAs
tau = T0/N

ETHz = np.zeros((N-1,), dtype=np.float64)#Поле THz
EFTHz = np.zeros((N-1,), dtype=np.float64)#Фурье - компоненты ТГц поля
ETHznorm = np.zeros((N-1,), dtype=np.float64)
n_0 = 0
n_1 = 0
n_2 = 0
t_0 = 0


#taue = np.float64(input('taue ='))
def f(t, n):
	return np.asarray([W0*math.exp(-t**2/taurate/taurate) - n[0]/tauc, -n[1]/taur + n[0]*e*n[2], -n[2]/taur + e/me*(Eb - n[1]/(nu*nInGaAs**2))], dtype = np.float64)
n = np.zeros((N,3), dtype = np.float64)
t = np.zeros((N,), dtype=np.float64)

n[0,:] = np.asarray([n_0, n_1, n_2], dtype = np.float64)
t[0] = t_0 #начальные условия

for i in range(0, N-1):
	nc = n[i,:]
	tc = t[i]
	k1 = f(tc, nc)
	k2 = f(tc + tau*0.5, nc + tau*k1*0.5)
	k3 = f(tc + tau*0.5, nc + tau*k2*0.5)
	k4 = f(tc + tau, nc + tau*k3)
	t[i+1] = t[i] + tau
	n[i+1,:] = n[i,:] + 1./6.*tau*(k1 + 0.5*k2 + 0.5*k3 + k4)
ne = n[:,0]
vel = n[:,2]
#print(t, ne)

ETHz = A*e/(c**2*z)*np.diff(ne*vel)*N/T0

max = ETHz[0]
pos = 0
for i in range(len(ETHz)):
	if ETHz[i] > max: max = ETHz[i]; pos = i
#print "max = ",max,", pos = ",pos

ETHzav = np.average(ETHz)
for i in range(len(ETHz)):
	ETHznorm[i] = ETHz[i] - ETHzav
EFTHz = np.absolute(np.fft.fft(ETHznorm))
n = EFTHz.size
timestep = T0/n
freq = np.fft.fftfreq(n, d=timestep)
max_a = np.amax(EFTHz)

plt.subplot(131)
plt.plot(t*10**12, ne)
pylab.xlabel('Time, ps')
pylab.ylabel('Concentration, sm(-3)')
pylab.ylim(ymin = 0)
pylab.xlim(xmin = -1, xmax = 10)

plt.subplot(132)
plt.plot(t[0:2000]*10**(12), ETHz[0:2000])
pylab.xlabel('Time, ps')
pylab.ylabel('Amplitude of THz field, a.u.')
#pylab.ylim(ymin = -1, ymax = 1)
pylab.xlim(xmin = -1, xmax = 10)

plt.subplot(133)
plt.plot(freq[0:2000]*10**(-12), EFTHz[0:2000])
#plt.yscale('log')
pylab.xlabel('Frequency, THz')
pylab.ylabel('FourierAmplitude of THz field, a.u.')
pylab.ylim(ymin = 0.001)
pylab.xlim(xmin = -0.1, xmax = 10)

#print "freq = ",freq,", FAmpl = ",EFTHz

f = open('Calculation.txt', 'wt')
for i in range(0,2000):
	#f.write(np.array_str(t[i]*10**12))
	a = np.array2string(t[i]*10**12, formatter = {'float_kind':lambda t:"%.1f" % t})+' '
	b = np.array2string(ETHz[i], formatter = {'float_kind':lambda ETHz:"%.1f" % ETHz})+'\n'
	f.write(a+b)
f.close()

plt.show()