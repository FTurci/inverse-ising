import numpy as np
from scipy import optimize
import h5py
import matplotlib.pyplot as plt
import tqdm
import cvxpy
import src.models as mod
from src.inverseising import pseudo_loglikelihood, pseudo_loglikelihood_grad, numgrad

produce_data = False
N = 200			

if produce_data:
	sk = mod.SK(N,1, outputfile =f'dataN{N}.hdf5')
	beta = 1.0
	mcsweeps =5*10**5

	for sweep in tqdm.tqdm(range(mcsweeps)):
		sk.sweep(beta)
		if sweep%10==0:
			sk.store()


	d = mod.load_data(f"dataN{N}.hdf5")
	J = mod.load_J(f"dataN{N}.hdf5")
	np.savez(f"dataN{N}.npz",d, J)


data = np.load(f'dataN{N}.npz')['arr_0'][::10]
print("Data", data.shape)
J =  np.load(f'dataN{N}.npz')['arr_1']

row, col = np.triu_indices(data.shape[1],k=1)
Ju = J[row,col]

print("J", Ju)

beta = 1.0

res = optimize.minimize(pseudo_loglikelihood,x0=np.ones(len(Ju)),args=(data,beta),
	jac = pseudo_loglikelihood_grad,	
	# method ='CG' , # 3 sec
	method ='Newton-CG' ,
	# method = 'Nelder-Mead'
	# method='L-BFGS-B' # 25 sec
	# method = 'Powell', # 90 sec
	# method = 'BFGS'  # 44 sec
	# method = 'TNC'	# >245 sec, does not converge
	# method = 'COBYLA'	#40 sec
	# method = 'SLSQP'	#22 sec
	# method = 'trust-constr' #112 sec
	options={ 'disp':True,'maxiter':20}#, 'gtol':1e-5}
	)




print('Ju',Ju)

print('Res', res.x)	
print(res)
plt.plot(Ju, res.x,'o')
plt.plot(Ju,Ju, 'k:')
plt.xlabel("Original J")
plt.ylabel("Inferred J")

plt.show()




