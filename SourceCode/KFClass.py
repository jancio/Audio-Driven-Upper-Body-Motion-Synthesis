#####################################################################################
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#####################################################################################
# Kalman Filter Class Module
# Usage: from KFClass import KFOnline
#####################################################################################
# Originally developed in my Part II Project "Replicating Human Facial Emotions and Head Movements on a Robot Avatar"
# The associated paper is titled "Automatic replication of teleoperator head movements and facial expressions on a humanoid robot"
#####################################################################################

import time
import numpy as np
#from cvxopt import matrix, solvers # not needed for Part III Project, 
# about constraints: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20030018910.pdf
import matplotlib.pyplot as plt

class KFOnline:
	'''
	Kalman Filter Class,
	state has dimension 3
	optionally can constrain the position
	with built in plotting
	'''

	# Specify:
	# process & observation noises
	# initial state & its covariance matrix
	# time step
	# optional: constraints!

	def __init__(self, dt, q, r, init_P_post, init_x_est_post, x_min=None, x_max=None):
		self.n_iter = 0								# iteration counter
		self.N = 3
		self.dt = dt
		self.x_sz = (self.N, 1)					# state vector dimensions
		self.P_sz = (self.N, self.N)

		self.x_min = x_min
		self.x_max = x_max

		self.x_est_prior = [] #np.zeros(x_sz)
		self.x_est_post =  [] #np.zeros(x_sz)
		self.P_prior = [] #np.zeros(P_sz)
		self.P_post = [] #np.zeros(P_sz)
		self.K = [] #np.zeros(x_sz)
		self.measurements = []

		self.Q = np.matrix([[0,0,0], [0,0,0], [0,0,q]])					# process variance / noise
		self.R = np.matrix([[r]])										# measurement/observation variance/noise
		self.F = np.matrix([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1]])	# transition matrix
		self.H = np.matrix('1 0 0')										# measurement matrix

		self.x_est_prior.append(np.zeros(self.x_sz))
		self.x_est_post.append(np.zeros(self.x_sz))
		self.P_prior.append(np.zeros(self.P_sz))
		self.P_post.append(np.zeros(self.P_sz))
		self.K.append(np.zeros(self.x_sz))

		self.P_post[0] = init_P_post 			#np.matrix('1 0 0; 0 1 0; 0 0 1')
		self.x_est_post[0] = init_x_est_post 	#np.matrix('0; 0; 0')

	# Measurement = yaw OR pitch
	def update(self, measurement, dt_new=None):
		self.n_iter += 1 # instead of k
		self.x_est_prior.append(np.zeros(self.x_sz))
		self.x_est_post.append(np.zeros(self.x_sz))
		self.P_prior.append(np.zeros(self.P_sz))
		self.P_post.append(np.zeros(self.P_sz))
		self.K.append(np.zeros(self.x_sz))
		self.measurements.append(measurement)
		if dt_new == None:
			F = self.F
		else:
			F = np.matrix([[1, dt_new, dt_new*dt_new/2], [0, 1, dt_new], [0, 0, 1]])	# actual transition matrix

		# time update (prediction)
		self.P_prior[self.n_iter] = F*self.P_post[self.n_iter-1]*F.T + self.Q 
		self.x_est_prior[self.n_iter] = F*self.x_est_post[self.n_iter-1]
		# measurement update (correction)
		self.K[self.n_iter] = self.P_prior[self.n_iter]*self.H.T / (self.H*self.P_prior[self.n_iter]*self.H.T + self.R)[0,0]
		self.x_est_post[self.n_iter] = self.x_est_prior[self.n_iter] + self.K[self.n_iter]*(measurement - self.H*self.x_est_prior[self.n_iter])[0,0]
		self.P_post[self.n_iter] = (np.identity(self.N) - self.K[self.n_iter]*self.H)*self.P_prior[self.n_iter]

		# Constrain the position naively
		''' very simple without adjustment of velocity and acceleration
		if(self.x_min != None):	
			self.x_est_post[self.n_iter][0,0] = max(self.x_est_post[self.n_iter][0,0], self.x_min[0])

		if(self.x_max != None):
			self.x_est_post[self.n_iter][0,0] = min(self.x_est_post[self.n_iter][0,0], self.x_max[0])
		'''

		# Constrain the position and velocity - quadratic programming
		if self.x_min != None and self.x_max != None:	
			# inverse of state covariance matrix => max probability estimate subject to constraints
			W = self.P_post[self.n_iter].I

			P = 2*matrix(np.array(W, dtype='d'))	
			q = matrix(np.array( -2 * W.T * self.x_est_post[self.n_iter] , dtype='d'))
			#G = matrix(np.array([ [1,0,0], [-1,0,0] ], dtype='d'))							# if only position is constrained
			#h = matrix(np.array([self.x_max[0], -self.x_min[0]], dtype='d'))				# if only position is constrained
			G = matrix(np.array([ [1,0,0], [-1,0,0], [0,1,0], [0,-1,0] ], dtype='d'))
			h = matrix(np.array([self.x_max[0], -self.x_min[0], self.x_max[1], -self.x_min[1]], dtype='d'))

			# suppressed output from QP solver
			solvers.options['show_progress'] = False
			sol = solvers.qp(P,q,G,h)
			# sol['x'] = x = argmin( x.T*W *x - 2*x_est_post.T*W*x )

			# if optimal solution was found, then adjust the state estimate x_est_post
			if sol['status'] == 'optimal':
				#if np.array(sol['x'])[1,0] != self.x_est_post[self.n_iter][1,0]:
				#	print np.array(sol['x'])[1,0]
				#	print self.x_est_post[self.n_iter][1,0]
				self.x_est_post[self.n_iter] = np.array(sol['x'])
			else:
				print "not optimal ... !"

		# equivalent to getLastEstState():
		return (self.x_est_post[self.n_iter][0,0], self.x_est_post[self.n_iter][1,0], self.x_est_post[self.n_iter][2,0])

	def updateAll(self, measurements):
		for m in measurements:
			self.update(m)

	def getEstStateAsArray(self):
		if self.n_iter == 0:
			return np.empty(0)
		return np.asarray(self.x_est_post[1:])[:,:,0]

	def getEstPositionAsArray(self):
		if self.n_iter == 0:
			return np.empty(0)
		return np.asarray(self.x_est_post[1:])[:,0,0]

	def getEstVelocityAsArray(self):
		if self.n_iter == 0:
			return np.empty(0)
		return np.asarray(self.x_est_post[1:])[:,1,0]

	def getEstAccelerationAsArray(self):
		if self.n_iter == 0:
			return np.empty(0)
		return np.asarray(self.x_est_post[1:])[:,2,0]

	def getLastEstState(self):
		return self.x_est_post[self.n_iter][:,0]

	def getLastEstPosition(self):
		return self.x_est_post[self.n_iter][0,0]

	def getLastEstVelocity(self):
		return self.x_est_post[self.n_iter][1,0]

	def getLastEstAcceleration(self):
		return self.x_est_post[self.n_iter][2,0]

	# Plot results of all 3 states ...
	def plotResults(self, titles, ylabels, xlabel='iteration', show=True):

		plt.figure()
		t = range(self.n_iter)

		for i in range(len(titles)):

			plt.subplot(len(titles), 1, i+1)
			if i == 0:
				plt.scatter(t, self.measurements, c='red', marker='+', label='measured')	# measured
			plt.plot(t, self.getEstStateAsArray()[:,i], 'x-', label='filtered')				# filtered
			if(self.x_min != None and i < len(self.x_min)):	
				plt.axhline(y=self.x_min[i], color='green', label='min/max constraints')	# constraints
			if(self.x_max != None and i < len(self.x_max)):	
				plt.axhline(y=self.x_max[i], color='green')
			plt.xlabel(xlabel)
			plt.ylabel(ylabels[i])
			plt.title(titles[i])
			plt.legend()
		if show:
			plt.show()

########################################################################################################################################
# Test:
########################################################################################################################################
'''

inFile = "./hpData1.dat"
measurements = np.genfromtxt(inFile, usecols = (0, 1))

mykf_Y = KFOnline(dt=1.0/30, q=0.5, r=0.01, init_P_post=np.matrix('1 0 0; 0 1 0; 0 0 1'), init_x_est_post=np.matrix('0; 0; 0'))
mykf_P = KFOnline(dt=1.0/30, q=0.5, r=0.01, init_P_post=np.matrix('1 0 0; 0 1 0; 0 0 1'), init_x_est_post=np.matrix('0; 0; 0'))
for i, m in enumerate(measurements, start=1):
	mykf_Y.update(m[0], 1.0/(i*10))				# to test changing dt over time
	mykf_P.update(m[1], 1.0/(i*10))
	# mykf_Y.update(m[0])
	# mykf_P.update(m[1])

xlabels = ['angle (rad)', 'angular velocity (rad.s-1)', 'angular acceleration (rad.s-2)']
mykf_Y.plotResults(['Yaw angle', 'Yaw angular velocity', 'Yaw angular acceleration'], xlabels, show=False)
mykf_P.plotResults(['Pitch angle', 'Pitch angular velocity', 'Pitch angular acceleration'], xlabels)
'''
########################################
#Test all:
'''
mykf_Y = KFOnline(dt=1.0/30, q=0.5, r=0.01, init_P_post=np.matrix('1 0 0; 0 1 0; 0 0 1'), init_x_est_post=np.matrix('0; 0; 0'))
mykf_P = KFOnline(dt=1.0/30, q=0.5, r=0.01, init_P_post=np.matrix('1 0 0; 0 1 0; 0 0 1'), init_x_est_post=np.matrix('0; 0; 0'))
inFile = "./hpData1.dat"
measurements = np.genfromtxt(inFile, usecols = (0, 1))
mykf_Y.updateAll(measurements[:,0])
mykf_P.updateAll(measurements[:,1])

xlabels = ['angle (rad)', 'angular velocity (rad.s-1)', 'angular acceleration (rad.s-2)']
mykf_Y.plotResults(['Yaw angle', 'Yaw angular velocity', 'Yaw angular acceleration'], xlabels, show=False)
mykf_P.plotResults(['Pitch angle', 'Pitch angular velocity', 'Pitch angular acceleration'], xlabels)
'''

###############################################
# Test constrained:
'''
inFile = "./hpData1.dat"
measurements = np.genfromtxt(inFile, usecols = (0, 1))

# Constraints on Angle, Angular velocity, Angular acceleration; (radians)
x_yaw_max = np.array([2.0857, 8.26797])
x_yaw_min = np.array([-2.0857, -8.26797])

x_pitch_max = np.array([0.200015, 7.19407, 100.])
x_pitch_min = np.array([-0.330041, -7.19407, -100.])


# Initialize Kalman Filter; with constraints (optional)
mykf_Y = KFOnline(dt=1.0/30, q=0.5, r=0.01, init_P_post=np.matrix('1 0 0; 0 1 0; 0 0 1'), init_x_est_post=np.matrix('0; 0; 0'), x_min=x_yaw_min, x_max=x_yaw_max)
mykf_P = KFOnline(dt=1.0/30, q=0.5, r=0.01, init_P_post=np.matrix('1 0 0; 0 1 0; 0 0 1'), init_x_est_post=np.matrix('0; 0; 0'), x_min=x_pitch_min, x_max=x_pitch_max)
mykf_P_noC= KFOnline(dt=1.0/30, q=0.5, r=0.01, init_P_post=np.matrix('1 0 0; 0 1 0; 0 0 1'), init_x_est_post=np.matrix('0; 0; 0'))

mykf_Y.updateAll(measurements[:,0])

tStart = time.time()
mykf_P_noC.updateAll(measurements[:,1])
print "Time without constraints: %f" %(time.time() - tStart)

tStart = time.time()
mykf_P.updateAll(measurements[:,1])
print "Time with constraints: %f" %(time.time() - tStart)		# cca 10 times slower !!!

xlabels = ['angle (rad)', 'angular velocity (rad.s-1)', 'angular acceleration (rad.s-2)']
mykf_Y.plotResults(['Yaw angle', 'Yaw angular velocity', 'Yaw angular acceleration'], xlabels, show=False)
mykf_P.plotResults(['Pitch angle', 'Pitch angular velocity', 'Pitch angular acceleration'], xlabels)
'''