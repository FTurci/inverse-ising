import numpy as np

""" Based on PRL 108, 090201 (2012)"""
count = 0

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
	# print (f.sum())
	# print(Jtriu, Jtriu.min())
	return f.sum()

def pseudo_loglikelihood_grad(Jtriu, spin=None, beta=None):
	global count
	print(count,"evaluation")
	N = spin.shape[1]
	B = spin.shape[0]

	# require Jij = Jji
	J = np.zeros((N,N))
	row, col = np.triu_indices(N,k=1)

	J[row,col] = Jtriu
	J += J.T

	# print(J)
	prod = -2*beta*spin[:,:,None]*spin[:,None,:]*J

	# print(prod)
	idx = (J!=0).astype(int)
	p = 1./(1+np.exp(prod))

	dpdj  = 2.0*beta*(spin[:,:,None]*spin[:,None,:]*idx)*p**2*np.exp(prod)


	analitic = np.mean(1/p*dpdj, axis=0)
	# print("analitic shape",analitic.shape)
	print("...ended")
	
	count+=1

	return -2*analitic[row,col]

def numgrad(Ju, delta,spin=None, beta=None):
	return	(pseudo_loglikelihood(Ju+delta, spin, beta)-pseudo_loglikelihood(Ju, spin, beta))/delta