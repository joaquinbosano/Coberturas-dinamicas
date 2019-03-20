
from dataclass import Dataholder
import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import pykalman as pykal
import quandl

#leer de base
futures = pd.read_pickle( "base.pkl")

spot = futures.iloc[:,0]
log_spot = np.log(spot)
log_spot.shape
my_tuple = pd.DataFrame(futures).shape
my_tuple
num_contracts = my_tuple[1]
num_days = my_tuple[0]

log_futures = np.log(futures)


# valores de inicialización

alpha=1.49
sigma_spot = 0.286
sigma_delta = 0.145
rho = 0.3
Lambda =  0.157
mu =0.0115
m = 0.2
r = 0.03
costo_fijo = 0.05
k_2 = alpha-rho*sigma_spot*sigma_delta
k_1 = np.sqrt(k_2**2+2*sigma_delta**2)




months = np.arange(num_contracts)
days = 30/360
Tau = np.array(months*days)
i = 0
B = []
Tau

for t in Tau:
    print(2*(1-np.exp(-k_1*t)) / (k_1+k_2+np.exp(-k_1*t)*(k_1-k_2)))


for t in Tau:
    B.append((2*(1-np.exp(-k_1*t)) / (k_1+k_2+np.exp(-k_1*t)*(k_1-k_2))))


B

B_neg = []
for elem in B:
    B_neg.append(-elem)
B_neg

C = np.ones (num_contracts)
obs = np.vstack((C,B_neg))

obs.shape
obs

tiempo = np.arange(num_days)
t = np.arange(num_days)
delta_t = 1/len(t)


#matriz de transicion
transition_M= np.array([[1, -delta_t*(1+0.5*sigma_spot**2)], [0,1+np.exp(-alpha*delta_t)]])

transition_M


#offset transicion
transition_offset = np.array([mu*delta_t, m*(1-np.exp(-alpha*delta_t))])


d = np.zeros(num_contracts)
j = 0

for mat in Tau:
    d[j] = (r+costo_fijo) * mat + (Lambda - alpha * m ) * (2/(k_1*(k_1-k_2))) * (np.log((k_1+k_2+np.exp(-k_1 * mat) * (k_1 -k_2))/(2*k_1)))
    j += 1
d
d.shape = (num_contracts,)

futures.iloc[1,:]

serie_sintetica = []

for t in range(len(tiempo)-1):
    serie_sintetica.append(np.abs(r+costo_fijo-12*np.log(futures.iloc[t,1]/futures.iloc[t,0])))

serie_sintetica
len(serie_sintetica)
len(tiempo)
varianza_condicional = []

for t in range(len(serie_sintetica)-1):
    if t > 0:
        varianza_condicional.append(m*((sigma_delta ** 2) / 2 * alpha) * (1- np.exp(-alpha * delta_t))**2 + serie_sintetica[t-1] * ((sigma_delta ** 2) / alpha) * (np.exp(-alpha*delta_t)-np.exp(-2*alpha*delta_t)))

varianza_condicional[1]
len(serie_sintetica)
len(varianza_condicional)
W = []

for t in range(len(varianza_condicional)-1):
    a= sigma_spot * delta_t * serie_sintetica[t]
    b= sigma_spot * rho * np.sqrt(varianza_condicional[t] * serie_sintetica[t])
    c = varianza_condicional[t]
    W.append(np.array([[a,b],[b,c]]))

W


# V= covarianza de las observaciones
V = np.identity(2)

measurements = log_futures
measurements.head()
measurements.iloc[:,0]

obs.T.shape
d.shape = (7,)
transition_M.shape
W[0].shape
num_contracts
transition_offset.shape


kf_2= KalmanFilter(n_dim_obs = num_contracts,
n_dim_state=2,
observation_matrices = obs.T,
transition_offsets = transition_offset,
transition_matrices = transition_M,
observation_offsets = d,
transition_covariance = W[0],
em_vars= ["observation_covariance", "initial_state_mean", "initial_state_covariance",
])



kf_2.filter(measurements)

state_means, state_covs = kf_2.filter(measurements)

np.save("cov_modeloCIR_crudo" ,state_covs)



t = np.arange(num_days)
t2 = np.arange(log_spot.shape[0])


state_means.shape
state_covs.shape
#Este es el más razonable"
#Ok, lo que estoy estimando con el filtro es el spot, no los futuros, ver schwartz pagina 897
log_futures.iloc[:,0]
t


# Guardar los graficos del modelo

for iter in range(num_contracts):
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(t, state_means[t,0]+ state_means[t,1]*B_neg[iter]+d[iter])
    plt.plot(t2, measurements.iloc[:,iter])

    plt.legend(["Estimador filtrado", "F_" + str(iter)])
    plt.xlabel('Tiempo')
    plt.ylabel('Log Precio')
    plt.savefig("crudoCIR"+str(iter)+".png" )
