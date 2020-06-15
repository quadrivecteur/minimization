import numpy as np
import matplotlib.pyplot as plt
import math

import keras
from keras import layers, models, optimizers

from scipy import integrate
from scipy import interpolate
from scipy.stats import norm

import time



def energy_compute (abscissa,fonction):
  hbar = 1
  omega = 1
  m = 1
  a = -5.
  b = 5.

  #interpolations
  tck_true = interpolate.splrep(abscissa, fonction, k=3, s=0)                                    #W.F.
  tck_true_carre = interpolate.splrep(abscissa, fonction*fonction, k=3, s=0)                     #W.F. squared
  tck_true_x = interpolate.splrep(abscissa, abscissa*abscissa*fonction*fonction, k=3, s=0)       #W.F. squared*x^2
  der_true = interpolate.splev(abscissa, tck_true, der=1)                                        #W.F. derivative
  tck_true_der = interpolate.splrep(abscissa,der_true*der_true, k=3,s=0)                         #W.F. derivative spline 1000
  int_true_carre = interpolate.splint(a,b,tck_true_carre)                                        #integral of W.F. squared
  int_true_x = interpolate.splint(a,b,tck_true_x)                                                #integral of W.F. squared*x^2 (<x^2>)
  int_true_der = interpolate.splint(a,b,tck_true_der)                                            #integral of derivative squared
  #energy
  Energy = ((-pow(hbar,2)/(2*m))*(fonction[-1]*der_true[-1]-fonction[0]*der_true[0] 
                             - int_true_der) + 0.5*m*omega*int_true_x ) / int_true_carre
  return Energy


def normalization (abscissa,function):
  a = -5.
  b = 5.

  tck_true_carre = interpolate.splrep(abscissa, function*function, s=0)             #W.F. squared
  int_true_carre = interpolate.splint(a,b,tck_true_carre)                           #integral of W.F. squared
  function = function*pow(1/int_true_carre,1/2)                                     #new normalized function

  return function





hbar = 1
omega = 1
m = 1
pts = 100
a = -5.
b = 5.
x = a
h = 10/pts
linx = np.linspace(a,b,pts)





#reference wave function
reference = np.zeros_like(linx, dtype=float)
for i in range(0,pts):
  reference[i] = pow(m*omega/(math.pi*hbar),0.25)*math.exp(-m*omega*(pow(x,2))/(2*hbar))
  x+=h
#now symmetric
for j in range(0,pts):
    reference[j] = (reference[j]+reference[pts-1-j])/2
    reference[pts-1-j] = reference[j]
#now normalized
reference = normalization(linx,reference)
#its energy
energy_ref = energy_compute(linx,reference)








#constant wave function (first target)
wave = np.ones_like(linx)
#normalized
wave = normalization (linx,wave)
#its energy
energy_wave = energy_compute(linx,wave)
#copy for plot at the end
first_target = wave



time1 = time.clock()


#INITIALIzATION OF NEURAL NETWORK

fits = 2000 #how many iterations we want

model = models.Sequential([
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(1), # no activation -> linear function of the input
])
model.summary()
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse',optimizer=opt)



for i in range(0,fits):

  #one training
  model.fit(linx,wave,epochs=1,batch_size=50,verbose=0)
  #prediction after one training
  predictions = model.predict(linx)

  #now positive, symetric and normalized, then its energy
  preds = np.abs(predictions.reshape(-1))
  for j in range(0,pts):
    preds[j] = (preds[j]+preds[pts-1-j])/2
    preds[pts-1-j] = preds[j]
  preds = normalization(linx,preds)

  energy_preds = energy_compute(linx,preds)

  #we choose the function with the lowest energy
  if (energy_preds < energy_wave):
    wave = preds
    energy_wave = energy_preds
    print('fit #',i+1)
    print('Energy = ',energy_preds)


  #go back to one training






print('')
print('Energy of the reference = ',energy_ref)
print('Energy found = ',energy_wave)
print('CPU time = ',time.clock()-time1)


plt.xlabel('x')
plt.ylabel('y')
plt.plot(linx,first_target,marker='.',c='darkgrey',label = 'first target',linestyle="None")
plt.plot(linx,reference,marker='.',c='deepskyblue',label = 'reference',linestyle="None")
plt.plot(linx,wave,marker='.',c='r',label = 'end result',linestyle="None")
plt.legend()
plt.savefig('minimization_01.pdf')