import numpy as np

""" Based on PRL 108, 090201 (2012)"""

def pseudo_loglikelihood(Jtriu, spin=None, beta=None):
	N = spin.shape[1]
	B = spin.shape[0]

	# require Jij = Jji
	J = np.zeros((N,N))
	row, col = np.triu_indices(N,k=1)

	J[row,col] = Jtriu
	J += J.T

 	# note: -log 1/P = log P 
	f = np.mean(np.log(
			1+np.exp(
				-2*beta*(spin@J)*spin)
			),
		axis=0
		)
	
	assert len(f)==N, 'Wrong dimensions'
	return f.sum()


