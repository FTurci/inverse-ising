import numpy as np
from scipy import sparse
import h5py


def load_J(filename):
	infile = h5py.File(filename,'r')
	J = np.array(infile['J'])
	return J

def load_data(filename):
	infile = h5py.File(filename,'r')
	
	confkeys = list(infile['configurations'].keys())
	B = len(confkeys)

	spin = np.array(infile['configurations'][confkeys[0]])
	N = len(spin)

	print("N =",N,"B =",B)

	data = np.zeros((B,N))
	for i,k in enumerate(confkeys):
		data[i] = np.array(infile['configurations'][k])

	return data

class SK:
	def __init__(self, N, p, outputfile=None, inputfile=None):
		self.N = N
		self.p = p
		self.c = p*N
		self.t = 0
		self.nsweep = 0


		if inputfile==None:
			self.spin = 2*np.random.randint(0,2,N)-1
		
			triu_N = int(N*(N-1)/2)
			connection  = np.random.uniform(0,1, size=triu_N)
			random_interaction = np.random.normal(0,1./self.c, size=triu_N)*(connection<p)

			row, col = np.triu_indices(N,k=1)

			assert len(row)==triu_N, "Incorrect initialisation."

			triuJ = sparse.coo_matrix((random_interaction, (row, col)), shape=(N, N))
			# J is symmetric
			self.Jsparse = triuJ+triuJ.T
			self.J = self.Jsparse.todense()

		else:
			self.infile = h5py.File(inputfile,'r')
			self.J = np.array(self.infile['J'])
			
			confkeys = np.sort(np.array(list(self.infile['configurations'].keys()), dtype=float) )

			self.spin = np.array(self.infile['configurations'][str(confkeys[-1])])
			

		if outputfile!=None:
			self.outputfile = outputfile
			self.outfile = h5py.File(outputfile,'w')
			self.data = self.outfile.create_dataset('J', data=self.J)
			self.confs =self.outfile.create_group('configurations')

	
			
	def energy_at(self, i):
		return float(self.spin[i]*(np.dot(self.J[i],self.spin)))

	def step(self,beta=None):

		pick = 	np.random.randint(self.N)
		neighbours =  -np.dot(self.J[pick], self.spin) # check the sign!
		delta = -2*self.spin[pick]*neighbours

		if delta<0:
			self.spin[pick]*=-1
		elif np.random.uniform(0,1) < np.exp(-beta*delta):
			self.spin[pick]*=-1
		self.t += 1

	def sweep(self,beta=None):
		for i in range(self.N):
			self.step(beta=beta)
		self.nsweep = self.t/self.N

	def store(self):
		assert self.outputfile != None,'Output file missing'
		self.confs.create_dataset(str(self.nsweep),data=self.spin)

