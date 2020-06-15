import numpy as np
import matplotlib.pyplot as plt
import math

import keras
from keras import layers, models, optimizers

from scipy import integrate
from scipy import interpolate
from scipy.stats import norm

import time



def calcul_energie (abscisse,fonction):
  hbar = 1
  omega = 1
  m = 1
  a = -5.
  b = 5.

  #Calcul des interpolations
  tck_true = interpolate.splrep(abscisse, fonction, k=3, s=0)                        		 #F.O. fonction d'onde
  tck_true_carre = interpolate.splrep(abscisse, fonction*fonction, k=3, s=0)             	 #F.O. module carré
  tck_true_x = interpolate.splrep(abscisse, abscisse*abscisse*fonction*fonction, k=3, s=0)       #F.O. <x^2>
  der_true = interpolate.splev(abscisse, tck_true, der=1)                   			 #F.O. dérivée
  tck_true_der = interpolate.splrep(abscisse,der_true*der_true, k=3,s=0)         		 #F.O. dérivée spline 100
  int_true_carre = interpolate.splint(a,b,tck_true_carre)               			 #F.O. module carré
  int_true_x = interpolate.splint(a,b,tck_true_x)                       			 #F.O. <x^2>
  int_true_der = interpolate.splint(a,b,tck_true_der)                   			 #F.O. derivée carré
  #Calcul de l'énergie
  Energie = ((-pow(hbar,2)/(2*m))*(fonction[-1]*der_true[-1]-fonction[0]*der_true[0] 
                             - int_true_der) + 0.5*m*omega*int_true_x ) / int_true_carre
  return Energie


def normalisation (abscisse,fonction):
  tck_true_carre = interpolate.splrep(abscisse, fonction*fonction, s=0)             #F.O. module carré
  int_true_carre = interpolate.splint(a,b,tck_true_carre)                           #F.O. module carré
  fonction = fonction*pow(1/int_true_carre,1/2)

  return fonction





hbar = 1
omega = 1
m = 1
pts = 100
a = -5.
b = 5.
x = a
h = 10/pts
linx = np.linspace (a,b,pts)





#vraie fonction d'onde
vraie_onde = np.zeros_like(linx, dtype=float)
for i in range(0,pts):
  vraie_onde[i] = pow(m*omega/(math.pi*hbar),0.25)*math.exp(-m*omega*(pow(x,2))/(2*hbar))
  x+=h
#symétrisation
for j in range(0,pts):
    vraie_onde[j] = (vraie_onde[j]+vraie_onde[pts-1-j])/2
    vraie_onde[pts-1-j] = vraie_onde[j]
#normalisation
vraie_onde = normalisation(linx,vraie_onde)
#calcul de son énergie
energie_cible = calcul_energie(linx,vraie_onde)








#fonction d'onde aléatoire (entre 0 et 1)
#onde = np.random.rand(pts)
onde = np.ones_like(linx)
"""
#symétrisation
for j in range(0,pts):
    onde[j] = (onde[j]+onde[pts-1-j])/2
    onde[pts-1-j] = onde[j]
"""
#normalisation
onde = normalisation (linx,onde)
#calcul de son énergie
energie_onde = calcul_energie(linx,onde)
#copie
onde1=onde



temps = time.time()


#INITIALISATION DU MODEL

batch=50
runs = 2000 #nombre de fits

model = models.Sequential([
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(100, input_shape=(1,), activation='relu'),
  layers.Dense(1), # no activation -> linear function of the input
])
#model.summary()
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse',optimizer=opt)



"""
for i in range(0,runs):

  #fit de l'onde
  model.fit(linx,onde,epochs=1,batch_size=batch,verbose=0)
  predictions = model.predict(linx)
  #symétrisation et valeur absolue de la prédiction + normalisation
  preds = np.abs(predictions.reshape(-1))
  for j in range(0,pts):
    preds[j] = (preds[j]+preds[pts-1-j])/2
    preds[pts-1-j] = preds[j]
  preds = normalisation(linx,preds)
  #calcul des énergies
  energie_preds = calcul_energie(linx,preds)

  #sélection de l'onde avec l'énergie la plus faible
  if (energie_preds < energie_onde):
    onde = preds
    energie_onde = energie_preds
    print('fit n°',i+1)
    print('Energie onde = ',energie_preds)
    
    keras.backend.clear_session()
    model = models.Sequential([
      layers.Dense(100, input_shape=(1,), activation='relu'),
      layers.Dense(100, input_shape=(1,), activation='relu'),
      layers.Dense(100, input_shape=(1,), activation='relu'),
      layers.Dense(100, input_shape=(1,), activation='relu'),
      layers.Dense(1), # no activation -> linear function of the input
    ])
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=opt)
    
  if energie_onde < 0.6 :
    print('_____________________________')
    print('           BREAK')
    print('_____________________________')
    break






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
"""



for i in range(0,2000):

  #fit de l'onde
  model.fit(linx,onde,epochs=1,batch_size=batch,verbose=0)
  predictions = model.predict(linx)

  #symétrisation et valeur absolue de la prédiction + normalisation
  preds = np.abs(predictions.reshape(-1))
  for j in range(0,pts):
    preds[j] = (preds[j]+preds[pts-1-j])/2
    preds[pts-1-j] = preds[j]
  preds = normalisation(linx,preds)

  energie_preds = calcul_energie(linx,preds)

  #sélection de l'onde avec l'énergie la plus faible
  if (energie_preds < energie_onde):
    onde = preds
    energie_onde = energie_preds
    print('fit n°',i+1)
    print('Energie onde = ',energie_preds)


temps2 = time.time()



"""
fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    data = open('stock.txt','r').read()
    lines = data.split('\n')
    xs = []
    ys = []
   
    for line in lines:
        x, y = line.split('\n') # Delimiter is comma    
        xs.append(float(x))
        ys.append(float(y))
   
    
    ax1.clear()
    ax1.plot(xs, ys)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Live graph with matplotlib')	
	
    
ani = animation.FuncAnimation(fig, animate, interval=1000) 
plt.show()
"""
print('Energie à trouver = ',energie_cible)
print('Energie trouvée = ',energie_onde)

plt.xlabel('x')
plt.ylabel('y')
plt.plot(linx,onde1,marker='.',c='darkgrey',label = 'first target',linestyle="None")
plt.plot(linx,vraie_onde,marker='.',c='deepskyblue',label = 'true ground state',linestyle="None")
plt.plot(linx,onde,marker='.',c='r',label = 'end result',linestyle="None")
plt.legend()
plt.savefig('run.pdf')
print('temps de calcul = ',temps2-temps)