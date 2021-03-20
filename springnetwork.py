import numpy as np
import matplotlib.pyplot as plt

"""
Create Your Own Spring Network Simulation (With Python)
Philip Mocz (2021) Princeton Univeristy, @PMocz

Simulate a system of connected springs
with Hooke's Law
"""
    

# convert subindex (i,j) to linear index (idx)
def sub2ind(array_shape, i, j):
    idx = i*array_shape[1] + j
    return idx


# calculate acceleration on nodes using hooke's law + gravity
def getAcc(pos, vel, ci, cj, springL, spring_coeff, gravity):
	# initialize
	acc = np.zeros(pos.shape)
	
	# Hooke's law: F = - k * displacement along spring
	sep_vec = pos[ci,:] - pos[cj,:]
	sep = np.linalg.norm( sep_vec, axis = 1)
	dL = sep - springL
	ax = - spring_coeff * dL * sep_vec[:,0] / sep
	ay = - spring_coeff * dL * sep_vec[:,1] / sep
	np.add.at(acc[:,0], ci, ax)
	np.add.at(acc[:,1], ci, ay)
	np.add.at(acc[:,0], cj, -ax)
	np.add.at(acc[:,1], cj, -ay)
	
	# gravity
	acc[:,1] += gravity
	
	return acc
	
	
# apply box boundary conditions, reverse velocity if outside box
def applyBoundary(pos, vel, boxsize):
	for d in range(0,2):
		is_out = np.where(pos[:,d] < 0)
		pos[is_out, d] *= -1 
		vel[is_out, d] *= -1 
		
		is_out = np.where(pos[:,d] > boxsize)
		pos[is_out, d] *= -1 
		vel[is_out, d] *= -1 
			
	return (pos, vel)


# simulation main loop
def main():
	""" N-body simulation """
	
	# Simulation parameters
	N         = 5      # Number of nodes per linear dimension
	t         = 0      # current time of the simulation
	dt        = 0.1    # timestep
	Nt        = 400    # number of timesteps
	spring_coeff = 40  # Hooke's law spring coefficient
	gravity   = -0.1   # strength of gravity
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	
	# construct spring nodes / initial conditions
	boxsize = 3
	xlin = np.linspace(1,2,N)

	x, y = np.meshgrid(xlin, xlin)
	x = x.flatten()
	y = y.flatten()
	
	pos = np.vstack((x,y)).T
	vel = np.zeros(pos.shape)
	acc = np.zeros(pos.shape)
	
	# add a bit of random noise
	np.random.seed(17)            # set the random number generator seed
	vel += 0.01*np.random.randn(N**2,2)
	
	# construct spring network connections
	ci = []
	cj = []
	#  o--o
	for r in range(0,Nlin):
		for c in range(0,Nlin-1):
			idx_i = sub2ind([Nlin, Nlin], r, c)
			idx_j = sub2ind([Nlin, Nlin], r, c+1)
			ci.append(idx_i)
			cj.append(idx_j)
	# o
	# |
	# o
	for r in range(0,Nlin-1):
		for c in range(0,Nlin):
			idx_i = sub2ind([Nlin, Nlin], r, c)
			idx_j = sub2ind([Nlin, Nlin], r+1, c)
			ci.append(idx_i)
			cj.append(idx_j)	
	# o
	#   \
	#     o
	for r in range(0,Nlin-1):
		for c in range(0,Nlin-1):
			idx_i = sub2ind([Nlin, Nlin], r, c)
			idx_j = sub2ind([Nlin, Nlin], r+1, c+1)
			ci.append(idx_i)
			cj.append(idx_j)	
	#     o
	#   /
	# o
	for r in range(0,Nlin-1):
		for c in range(0,Nlin-1):
			idx_i = sub2ind([Nlin, Nlin], r+1, c)
			idx_j = sub2ind([Nlin, Nlin], r, c+1)
			ci.append(idx_i)
			cj.append(idx_j)
	
	# calculate spring rest-lengths
	springL = np.linalg.norm( pos[ci,:] - pos[cj,:], axis = 1)
	
	
	# prep figure
	fig = plt.figure(figsize=(4,4), dpi=80)
	ax = fig.add_subplot(111)
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		vel += acc * dt/2.0
		
		# drift
		pos += vel * dt
		
		# apply boundary conditions
		pos, vel = applyBoundary(pos, vel, boxsize)
		
		# update accelerations
		acc = getAcc( pos, vel, ci, cj, springL, spring_coeff, gravity )
		
		# (1/2) kick
		vel += acc * dt/2.0
		
		# update time
		t += dt
		
		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.cla()
			plt.plot(pos[[ci, cj],0],pos[[ci, cj],1],color='blue')
			plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
			ax.set(xlim=(0, boxsize), ylim=(0, boxsize))
			ax.set_aspect('equal', 'box')
			ax.set_xticks([0,1,2,3])
			ax.set_yticks([0,1,2,3])
			plt.pause(0.001)
	    
	
	
	# Save figure
	plt.savefig('springnetwork.png',dpi=240)
	plt.show()
	    
	return 0
	


  
if __name__== "__main__":
  main()
