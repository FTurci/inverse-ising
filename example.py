import numpy as np
from scipy import optimize
import h5py
import matplotlib.pyplot as plt
import tqdm

import src.models as mod
from src.inverseising import pseudo_loglikelihood

produce_data = False

if produce_data:
	sk = mod.SK(10,1, outputfile ='data.hdf5')
	beta = 1.0
	mcsweeps =10**6

	for sweep in tqdm.tqdm(range(mcsweeps)):
		sk.sweep(beta)
		if sweep%10==0:
			sk.store()


	d = mod.load_data("data.hdf5")
	J = mod.load_J("data.hdf5")
	np.savez("data.npz",d, J)


data = np.load('data.npz')['arr_0']
J =  np.load('data.npz')['arr_1']

row, col = np.triu_indices(data.shape[1],k=1)
Ju = J[row,col]

print("J", Ju)

beta = 1.0

ps = pseudo_loglikelihood(Ju,spin=data,beta=beta)
print ('Pseduo logliklelihood', ps)

res = optimize.minimize(pseudo_loglikelihood,x0=np.ones(len(Ju)),args=(data,beta), 
	method='L-BFGS-B',
	)

print('Ju',Ju)
print('Res', res.x, res.fun)

plt.plot(Ju, res.x,'o')
plt.plot(Ju,Ju, 'k:')
plt.xlabel("Original J")
plt.ylabel("Inferred J")

plt.show()




