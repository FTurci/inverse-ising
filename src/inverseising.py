import numpy as np
from scipy import optimize
import h5py
import matplotlib.pyplot as plt
""" Based on PRL 108, 090201 (2012)"""

def pseudo_loglikelihood(Jtriu, spin=None, beta=None):
	
	N = spin.shape[1]
	B = spin.shape[0]
	
	# require Jij = Jji
	J = np.zeros((N,N))
	row, col = np.triu_indices(N,k=1)
	J[row,col] = Jtriu
	J += J.T

	f = np.mean(
		np.log(
			1+np.exp(
				-2*beta*(spin@J)*spin)
			),
		axis=0)
	
	assert len(f)==N, 'Wrong dimensions'
	return f.sum()

def load_data(filename):
	infile = h5py.File(filename,'r')
	J = np.array(infile['J'])
	
	confkeys = list(infile['configurations'].keys())
	B = len(confkeys)

	spin = np.array(infile['configurations'][confkeys[0]])
	N = len(spin)

	print("N =",N,"B =",B)

	data = np.zeros((B,N))
	for i,k in enumerate(confkeys):
		data[i] = np.array(infile['configurations'][k])

	return data


# print(d)
data = np.load('data.npz')['arr_0']
J = load_J('data.hdf5')
row, col = np.triu_indices(data.shape[1],k=1)
Ju = J[row,col]

beta = 1.0

ps = pseudo_loglikelihood(Ju,spin=data,beta=beta)
print ('Pseduo logliklelihood', ps)

res = optimize.minimize(pseudo_loglikelihood,x0=np.ones(len(Ju)),args=(data,beta), 
	method='L-BFGS-B'
	)

print('Ju',Ju)
print('Res', res.x, res.fun)

plt.plot(Ju, res.x,'o')
plt.plot(Ju,Ju, 'k:')
plt.xlabel("Original J")
plt.ylabel("Inferred J")
plt.show()