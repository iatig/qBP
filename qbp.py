########################################################################
#
#   qbp.py --- Plain BP for tensor networks
#   ===================================================================
#
#
#  Changelog
#  ----------
#
#  8-Feb-2024: Itai  Initial commit. Added the ket option.
#
#  25-Feb-2024: Itai  Fixed a bug in insideout_DL: removed a conj() 
#                     function on the BP message that was contracted to 
#                     the bra tensor.
#
#  2-Jun-2024: Itai 1. Make the outgoing messages in insideout_DL 
#                      explicitly hermitian
#                   2. Normalize by L2 norm the outgoing messages
#                   3. Change the way we calculate the err in the 
#                      BP loop --- to better match the blockBP alg.
#
# 12-Jun-2024: Itai Have a more strtictly defined behaviour in 
#                   the two cases: ket (quantum) , ketbra (classic).
#                   ket is for double layer and ketbra is a normal
#                   classical BP. This difference is apperant in:
#                   1.  Initialization: in ket mode, use random 
#                       PSD with trace=1
#                       In ketbra mode  use uniform 1
#                   2.  BP message normalization: in ket use trace=1
#                       whereas in ketbra use sum(m)=1.
#
#                   In addition, change the BP normalization part: outgoing
#                   messages are automatically normalized (each mode by
#                   its own way). Then BP error is calculated using a 
#                   normalized L_1 norm (in both cases).
#
# 12-Jun-2024: Itai Removed the use_ket flag in qbp. Now it is 
#                   automatically detected from comparing the shape
#                   of T_list[0] to len(e_list[0]).  Changed input
#                   parameter m_list -> inital_m. Now it can be 
#                   either a messages list, or 'R' (random) or 'U' 
#                   (uniform). Changed in function body edegs_list 
#                   -> e_list. 
#
# 12-Jun-2024: Itai Added the get_Bethe_free_energy() function. 
#                   Currently only works for the Classical case.
#
# 14-Jun-2024: Itai Updated the form of e_dict to (i,i_leg, j,j_leg)
#                   
# 14-Jun-2024: Itai Add the calc_e_dict to the module. Add an optional
#                   e_dict parameter to qbp() and use it to replace
#                   the vertices dictionary
#
# 18-Aug-2024: Itai get_Bethe_free_energy: now works also from quantum
#                   messages (matrices)
#
#
# 1-Apr-2025:  Itai In get_Bethe_free_energy, make sure M is complex 
#                   when taking log(M)
#
# 1-Apr-2025:  Itai Fix the truncation to real in get_Bethe_free_energy
#
# 20-Jun-2025: Itai changes in:
#              (1) get_Bethe_free_energy:
#                  - Instead of normalizing the messages by their 
#                    overlap, we directly add the log of this overlap to
#                    the Bethe free energy.
#                  - Made it more robust to edge cases where we attempt 
#                    to take the log or exponent of very large numbers
#
#              (2) qbp: add a flag normalzie_it that decides if to 
#                       normalize the outgoing messages. This happens 
#                       only if their norm sufficiently large 
#                       (either L_1 in classical case or L_2 in quantum case)
#
#
# 14-Aug-2025: Itai Many changes: 
#                   (1) rename insideout_ket -> insideout_DL
#                              inside_out -> insideout_SL
#                   (2) renamed modes in qbp: 'Q' -> DL, 'C' -> SL
#                   (3) changed normalization of BP messages:
#                       SL: normalize by L_1 and then by phase of sum
#                       DL: normalize by L_1 (nuc norm) and the by phase
#                           of trace.
#                   (4) Accordingly the distance is also L_1 in both
#                       cases (changed the DL to ord='nuc' so that it 
#                       uses the matrix L_1 norm).
#
#
# 17-Aug-2025: MAJOR change in qbp: changed the permute_order flat to 
#              the vorder parameter, which can now also be used to 
#              specify an arbitrary order of vertices. In addition, 
#              replaced all log/Log flags with elog.
#
#
# 18-Aug-2025: Added the cluster_qbp function, together with the
#              cluser_Bethe_free_energy. Currently, the cluster_qbp
#              only works in Single-Layer mode (SL) and supports only
#              'star' and 'loop' clusters.
#
# 18-Sep-2025: Add function adj_vert. Also added functionality to 
#              treat wcopy and xor tensors, which can be used as 
#              clusters in cluster_qbp for QEC decoding. This includes
#              adding this functionality to cluster_qbp, via the new  
#              functions insideout_xor, insideout_wcopy, and also the 
#              appropriate subfunctions in cluster_Bethe_free_energy. 
#              Also remove degenerate DL functionality from cluster_qbp.
#
#
# 18-Sep-2025: Added "r" to the comment prefactor at the beginning of 
#              each function.
#
# 13-Nov-2025: Condition the phase-normalization of each message by
#              requiring that the trace/sum of the message is bigger
#              then some threshold stored in TR_NORM_EPS. This way
#              we avoid dividing by zero.
#


import numpy as np

from numpy.linalg import norm
from numpy import sqrt, dot, vdot, tensordot, array, zeros, ones, conj, trace,\
	eye, log, pi


nscHgate = array([[1,1],[1,-1]])  # *Unscaled* Hadamard gate



#
# ---------------------------- calc_e_dict  ----------------------------
#

def calc_e_dict(e_list):
	r"""
	
	Given an edge structure in the form of e_list, calculate the edges
	dictionary. This is a dictionary in which the key is an edge. 
	
	The value of each edge e=(i,j) is a 4-taple: 
	
	                (i,leg_i, j,leg_j)
	                
	where leg_i, leg_j are the indices of the leg e in the tensors at i
	and j and i<j.
	
	
	"""
	
	e_dict = {}
	
	for j,es in enumerate(e_list):
		
		for leg_j,e in enumerate(es):
			
			if e in e_dict.keys():
				#
				# If the edge already exists in the dictionary, then this is the
				# second vertex --- then add it to form a 4 taple
				#   (i,leg_i, j,leg_j)
				#
				
				i, leg_i = e_dict[e]
				
				e_dict[e] = (i,leg_i, j,leg_j)
					
			else:
				#
				# If the edge does not exist --- we just add this vertex
				#
				e_dict[e] = (j, leg_j)

	return e_dict


#
# ------------------------   adj_vert   -------------------------
#
def adj_vert(v, e, e_dict):
	r"""
	
	Returns the vertex that is adjacent to vertex v along edge e
	
	"""
	
	i,i_leg, j, j_leg = e_dict[e]
	
	if i==v:
		return j
	else:
		return i



#
# ------------------  get_Bethe_free_energy   --------------------------
#
def get_Bethe_free_energy(m_list, T_list, e_list, e_dict):
	
	r"""
	
	Given a TN and a converged set of BP messages, calculate the 
	Bethe Free Energy F_bethe, which gives the BP approximation of the total
	contraction of the TN:
	
	Tr(TN) ~ e^{-F_bethe}
	
	The calculation is done according to Lemma~III.1 in arXiv:2402.04834v2
		
	Input Parameters:
	------------------
	m_list --- converged BP messages. This is a double-list such that
	           m_list[i][j]  is the converged i->j message.
	           
	T_list, e_list, e_dict --- TN structure.
	
	Output:
	--------
	F_bethe
	
	"""

	elog = False
	
	EPS    = 1e-15
	BIGLOG = 50
	
	#
	# Check if we're in quantum or classical mode
	#
	
	if len(T_list[0].shape)==len(e_list[0]):
		#
		# Classical mode --- there's no physical leg to the local tensor
		#
		mode = 'C'
	else:
		#
		# Quantum mode --- local tensor has a phyical leg
		#
		mode = 'Q'
	
	if elog:
		print(f"Entering get_Bethe_free_energy in {mode} mode")
		
	n = len(T_list)
	
	#
	# First, normalize the messages such that the inner product of 
	# the messages along an edge = 1
	#
	# Tr( m_{i\to j} \cdot m_{j\to i} ) = 1
	#
	# See Lemma III.1 in the blockBP paper, arXiv:2402.04834
	#
	# Alternatively, instead of normalizing the messages by their 
	# overlap, we directly add the log of this overlap to the Bethe 
	# free energy.
	#

		
	F_bethe = 0.0j
		
	for e in e_dict.keys():
		
		i,i_leg, j,j_leg = e_dict[e]

		
		if mode=='C':
			#
			# We're on classical mode; messages are vectors
			#
			msg_overlap = sum(m_list[i][j]*m_list[j][i])
		else:
			#
			# We're on quantum mode; messages are matrices
			#
			msg_overlap = trace( m_list[i][j]@(m_list[j][i]).T )
		
		#
		# We need to be careful when adding the log of the message-overlap.
		# It can be zero (or very close to zero), and it can also be 
		# negative. Therefore, we must work in complex numbers and also 
		# check the absolute value before taking the log.
		#
		nr = sqrt(abs(msg_overlap))
		
		if nr<EPS*max(norm(m_list[i][j]), norm(m_list[j][i]),1e-20):
			F_bethe += -BIGLOG
			
			if elog:
				print("Warnning: small message overlap in get_Bethe_free_energy!")
				print(f"message-overlap: {msg_overlap:.6g}")
				print(f"msg({i}->{j}) norm: {norm(m_list[i][j]):.6g}      " \
					f"msg({j}->{i}) norm: {norm(m_list[j][i]):.6g} " )
				print()
		else:
			try:
				F_bethe += log(msg_overlap.astype(np.complex128))
			except FloatingPointError:
				print("error: msg_overlap=",msg_overlap, "nr=", nr)
				print("message norms: ", norm(m_list[i][j]), norm(m_list[j][i]))
				exit(0)
					
		
	
	#
	# Run over all vertices and add up the log of the contraction of 
	# the normalized incoming messages with T_i
	#
	
	if mode=='C':
		#
		# Classical mode
		#
		for i in range(n):
			
			M = T_list[i]
			for e in e_list[i]:
				
				vi,i_leg, vj,j_leg = e_dict[e]
				
				if vi==i:
					j=vj
				else:
					j=vi
			
				M = tensordot(M, m_list[j][i],axes=([0],[0]))
			
			#
			# If M is too small, we just increase F_bethe by some
			# insane amount
			#
			if abs(M)<EPS*norm(T_list[i]):
				F_bethe += BIGLOG

				if elog:
					print("Warnning: small local contribution in get_Bethe_free_energy!")
					print(f"Local contrib at vertex {i}: {M:.6g}    "\
						f"norm(T[{i}]) = {norm(T_list[i]):.6g}")
					print()
			else:
				F_bethe += -log(M.astype(np.complex128))
				
			
	else:
		#
		# Quantum mode
		#
	
		for i in range(n):
			
			M = T_list[i]
			for e in e_list[i]:
				
				vi,i_leg, vj,j_leg = e_dict[e]
				
				if vi==i:
					j=vj
				else:
					j=vi
			
				M = tensordot(M, m_list[j][i],axes=([1],[0]))
				
			M = dot(M.flatten(), conj(T_list[i].flatten()))
			
			#
			# At this point, M is the contraction of the ket with all the 
			# ket-legs of the incoming messages. We now contract this with 
			# the bra
			#
			
			F_bethe = F_bethe - log(M.astype(np.complex128))
	
	#
	# Make the imaginary part minimal (rememeber that 2*pi multiples do
	# not matter because Tr(TN)=exp(-F_bethe) )
	#
	# Also, if we are very close the positive real axis, then probably
	# we're on a classical settings, and so just ignore the imaginary
	# part of the Bethe energy, and output a real number.
	#

	F_bethe_i = F_bethe.imag % (2*pi)
	
	if min(abs(F_bethe_i), abs(F_bethe_i-2*pi)) < EPS*abs(F_bethe):
		F_bethe = F_bethe.real
	else:
		F_bethe = F_bethe.real + 1j*F_bethe_i
		
	return F_bethe












#
# --------------------  cluster_Bethe_free_energy  ---------------------
#	
def cluster_Bethe_free_energy(m_list, T_list, e_list, e_dict, c_list):
	
	

	#
	# ~~~~~~~~~~~~~~~~~~~~   contract_external_legs 
	#
	def contract_external_legs(v0, v1, v2, bridges=[]):
			
		r"""
		
		Given 3 vertices v0,v1,v2 we focus on the tensor of v1. We contract
		it with all the incoming messages that are *not* coming from v0 or 
		v2 or any of the bridges. 
		
		The result is either a rank-2 tensor or a rank-3 tensor (if there's
		a bridge).
		
		We then permute the legs according to '02' or '0b2' where:
		0 --- leg connecting to v0
		b --- leg connecting to the bridge (if any)
		2 --- leg connecting to v2
		
		Input Parameters:
		-----------------
		v0,v1,v2       --- The indices of the vertices in the cluster, 
											 where v1 is the index of the tensor we contract
											 
		bridges        --- An optional list of internal edges
		
		
		Output:
		-------
		Either a degree 2 or 3 tensor M:
		degree 2: T_{i0,i2}
		degree 3: T_{i0,b,i2}
		
		
		"""
		
		M = T_list[v1]
		
		order = ''
		for e in e_list[v1]:
			
			if e in bridges:
				order = order + 'b'
				action = 'cycle'
			
			else:
			
				vi,i_leg, vj,j_leg = e_dict[e]
				
				if vi!=v1:
					vj=vi
					
				#
				# Now vj is the neighboring vertex
				#
				
				if vj in [v0,v2]:
					#
					# So vj is one of the legs in the loop
					#
					if vj==v0:
						order = order + '0'
					else:
						order = order + '2'
					
					action = 'cycle'
				else:
					action = 'contract'

			if action == 'cycle':
				#
				# Create a permutation that cycles the 1st leg to be last
				#
				k = len(M.shape) # how many legs
				sh = list(range(1,k))
				sh.append(0)

				M = M.transpose(sh)
				
			if action == 'contract':
				#
				# Its an external leg - so contract with the incoming message
				#

				M = tensordot(M, nm_list[vj][v1],axes=([0],[0]))
		
		#
		# At this point the legs of resultant tensor can be of the orders
		# '02', '20', '02b', 'b02', ...
		#
		# We want to bring it to either '02' or '0b2'
		#
		if len(order)==2:
			if order == '20':
				M = M.transpose()
		else:
			sh = [order.index('0'), order.index('b'), order.index('2')]
			M = M.transpose(sh)
			
		return M

	
	
	#
	# ~~~~~~~~~~~~~~~~~~~~~~~  cluster_wcopy
	#
	def cluster_wcopy(in_m_list, p=None):
		
		k = len(in_m_list)
			
		if p is None:
			m0, m1 = 1.0, 1.0
		else:
			m0, m1 = 1-p, p
		
		for i in range(k):
			m0 *= in_m_list[i][0]
			m1 *= in_m_list[i][1]
		
		tr = m0+m1
		
		return log(tr.astype(np.complex128))
			
		
	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cluster_xor
	#
	def cluster_xor(in_m_list, parity=0):
		
		Hm_list = [nscHgate@m for m in in_m_list]

		if parity==1:
			v = array([0,1])
			Hm_list.append(nscHgate@v)
		
			
		return cluster_wcopy(Hm_list) - log(2)

	
	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~  cluster_star
	# 
	def cluster_star(i):
		
		M = T_list[i]

		for e in e_list[i]:
				j = adj_vert(i, e, e_dict)
				M = tensordot(M, nm_list[j][i],axes=([0],[0]))
				
		
		return log(M.astype(np.complex128))
			
	


	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~  cluster_loop
	# 
	def cluster_loop(v_list):
		
		l=len(v_list)
		
		for i in range(l):
			v0 = v_list[(i-1)%l]
			v1 = v_list[i]
			v2 = v_list[(i+1)%l]
			
			M = contract_external_legs(v0,v1,v2)
			
			if i==0:
				loop_O = M
			else:
				loop_O = loop_O@M
				
		tr = trace(loop_O)
			
		return log(tr.astype(np.complex128))
			


	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~  cluster_ladder
	# 
	
	def cluster_ladder(v_list, bridges):
		
		l = len(v_list)
		
		i_R, i_L, M = -1, l, None

		#
		# We start by contracting the right branch
		#
		mode = 'R'
		
		while i_R+1 < i_L:
			
			if mode=='R':
				
				#
				# Update i_R and find the contracted tensor there
				#
				i_R += 1
				
				v0 = v_list[(i_R-1)%l]
				v1 = v_list[i_R]
				v2 = v_list[(i_R+1)%l]
				
				T_R = contract_external_legs(v0,v1,v2, bridges)
				
				#
				# Now contract it
				#
				if M is None:
					M = T_R
				elif len(T_R.shape)==2:
					M = M@T_R
				else:
					#
					# we've encountered a bridge. So switch to contracting the 
					# left branch until we meet the other side of the bridge
					#
					mode = 'L'
					
			if mode=='L':
				
				#
				# Start by updating the left index and getting the contracted
				# tensor there
				#
				i_L -= 1
				
				v0 = v_list[(i_L-1)%l]
				v1 = v_list[i_L]
				v2 = v_list[(i_L+1)%l]
				
				
				T_L = contract_external_legs(v0,v1,v2, bridges)
				
				if len(T_L.shape)==2:
					M = T_L@M
				else:
					#
					# we've reached a bridge. This means that T_R is also a 
					# degree-3 tensor. So now we need to contract the bridge
					# T_R, M, T_L 
					#
					# and then switch to the right branch
					#
					
					T_R = tensordot(M, T_R, axes=([1],[0]))
					M = tensordot(T_L,T_R, axes=([2,1],[0,1]))
					
					mode = 'R'
					
		tr = trace(M)
		return log(tr.astype(np.complex128))
		

	#~~~~~~~~~~~~~~~~~~~~~     Start of function      ~~~~~~~~~~~~~~~~~~~~
		

	Log = False
	
	EPS    = 1e-15
	BIGLOG = 50
		
	n = len(T_list)
	
	nm_list = [[None]*n for i in range(n)]
	
	#
	# First, renormalize the messages so that 
	#

	for e in e_dict.keys():
		
		i,i_leg, j,j_leg = e_dict[e]
		
		# by construction we have i<j

		msg_overlap = dot(m_list[i][j],m_list[j][i])
		nm_list[i][j] = m_list[i][j]/msg_overlap
		nm_list[j][i] = m_list[j][i]
	
	
	F_bethe = 0.0j
	
	
	for ci, c in enumerate(c_list):
		ctype, v_list, params = c
		
		
		if ctype=='star':
			v = v_list[0]
			F_bethe = F_bethe - cluster_star(v)
			
		if ctype=='loop':
			F_bethe = F_bethe - cluster_loop(v_list)
		
		if ctype=='ladder':
			F_bethe = F_bethe - cluster_ladder(v_list, params['bridges'])
			
			
		if ctype=='xor' or ctype=='wcopy':
			v = v_list[0]

			in_m_list = []
			for e in e_list[v]:
				vj = adj_vert(v,e, e_dict)
				in_m_list.append(nm_list[vj][v])
				
			if ctype=='xor':
				F_bethe = F_bethe - cluster_xor(in_m_list, params)
			else:
				F_bethe = F_bethe - cluster_wcopy(in_m_list, params)
	
	#
	# Make the imaginary part minimal (rememeber that 2*pi multiples do
	# not matter because Tr(TN)=exp(-F_bethe) )
	#
	# Also, if we are very close the positive real axis, then probably
	# we're on a classical settings, and so just ignore the imaginary
	# part of the Bethe energy, and output a real number.
	#

	F_bethe_i = F_bethe.imag % (2*pi)
	
	if min(abs(F_bethe_i), abs(F_bethe_i-2*pi)) < EPS*abs(F_bethe):
		F_bethe = F_bethe.real
	else:
		F_bethe = F_bethe.real + 1j*F_bethe_i
	
	
		
	return F_bethe
		
	

#
# --------------------------- cluser_qbp --------------------------------
#

def cluster_qbp(T_list, e_list, e_dict, c_list, \
	initial_m='U', max_iter=10000, delta=1e-6, damping=0.2, \
	corder='sequential'):

	r"""
	
	Runs a BP (belief Propagation) on a (closed) tensor network and 
	where the messages between vertices are calculated using clusters.
	
	Currently a cluster can be a single vertex, or a simple loop. 
	
	Given a cluster and a set of incoming messages to its vertices, the 
	every outgoing message  i->j of a vertex i in the cluster and an 
	external vertex j is calculated by:
	
	(a) Contract all incoming messages from external vertices to the 
	    cluster, apart from the vertex j
	    
	(b) Exactly contract all the remaining TN of the cluster.
	

	Currently, the function on works in single-layer (SL) mode.
	
	Except for this difference, the functions behaves like the ordinary
	BP (see qbp). We run a number of iterations. In each iteration we go 
	over the clusters in a certain order, and for each cluster we run the
	corresponding insideout function, which takes the incoming messages 
	of the cluster and calculates its outgoing messages.
	
	The iterations stop once we hit the upperbound (max_iter), or the
	messages of two previous generations are close enough to each other.
	
	
	Parameters:
	-----------
	
	T_list --- The list of tensors that make up the TN (like in ncon)
	
	e_list ---	The labels of the edges (legs) of each tensor. This
									is a list of lists. For each tensor T with k legs
									there is a list [e_0, e_1, ..., e_{k-1}] of the labels
									of the legs. The same label must appear in the other
									tensor with which the leg is contracted.
	               
									Each label must be a positive integer.
									
	e_dict   --- A complementary structure to e_list. It is a dictionary
	             where the keys are the edges labels, and the value
	             is a 4-taple (i,i_leg, j, j_leg) with i<j.
	             
	             It is completely derivable from e_list. If omitted, 
	             it is calculated at the begining of the function.
	             
	c_list   --- The list of clusters. Each cluster is a tuple
	             (ctype, vs, params), where:
	             
	               (-) ctype is a string:
	                   => 'star' --- a single vertex
	                   => 'loop' --- a simple loop
	             
	               (-) vs --- list of vertices that make up the cluster
	             
	               (-) params --- an optional dictionary of parameters
									
	               
	initial_m   --- An optional list of initial messages. It can also be 
	                a string 'U' meaning unifrom initialization (1/D 
	                vector in the classical case or Id/D matrix in the 
	                quantum case), or it can be 'R', in which case we use
	                random normalized vector vector in the classical case
	                and a random PSD with Tr=1 in the quantum case).
	
									
	max_iter ---		A upper bound for the maxmial number of iterations
	
	delta			---			The distance between two consequtive sets of messages
									after which the iteration stops (the messages are 
									considered converged)
									
	damping --- 		A possible damping parameter to facilitate convergence.
									The new message is obtained by:
									
									(1-damping)*new_m + damping*old_m
									
									
	corder ---      The order of the clusters in which BP is run; on each
	                round the algorithm go over all clusters according to
	                that order.
	                
	                There are 3 possibilities:
	                'sequential' --- go over v=0,1,2,3,...
	                'random'     --- picks a random order
	                actual list  --- in  this case the list contains the
	                                 order of the clusters.
	                  
	
	Output:
	-------
	
	The list of converged messages. This is a double list m_list[i][j]
	which holds the i->j message.

	
	
	"""
			
	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~  clegs
	# 
	
	def clegs(v0,v1,v2):
		
		r"""
		
		Given a sequence of vertices on the loop v0 -> v1 -> v2, the function
		calculates the relevant tensors for v1. These include:
		
		1. The transfer matrix T_{k0,k2} which is the contraction of all the
		   legs of v1 except for those of v0, v2
		   
		2. External tensors M_{k0, k2, k}, where k is a remaining external
		   leg (and all the other external legs are contracted).
		   
		the output is T and a list [(j0, M0), (j1, M1), ...]
		
		where j_i is the external vertex to which the external leg of Mi 
		belongs.
		
		
		"""
		
		M = T_list[v1]

		#
		# We first create a list of all incoming messages to v1, and 
		# note which one of them belongs to v0, v2, and which are external
		#
		
		in_msg_list = []
		for k,e in enumerate(e_list[v1]):
			
			vi,i_leg, vj,j_leg = e_dict[e]
			
			if vi!=v1:
				vj=vi
				
			#
			# Now vj is the neighboring vertex
			#
			
			if vj==v0:
				v0_loc = k
			elif vj==v2:
				v2_loc = k
			else:
				#
				# Its an external, so add its incoming message to the list
				#
				
				in_msg_list.append((k,vj,m_list[vj][v1]))

		#
		# Now contract the external legs to their incoming messages, 
		# forming the M_i 3-leg tensors.
		#
		# At this point we only support the cases where the number of 
		# external legs is 0, 1, 2
		#
		
		ext_n=len(in_msg_list)
		
		if ext_n==0:
			#
			# No external legs
			#
			if v2_loc<v0_loc:
				#
				# Make sure that the legs order in M is (k0, k2)
				#
				M = M.transpose()
			
			return M, []
			
		if ext_n==1:
			#
			# One external leg
			#
			
			k,j,m = in_msg_list[0]
			
			#
			# Move the external leg to the end and make sure that the k0,k2
			# legs are ordered properly
			#

			perm = (v0_loc,v2_loc,k)
			M = M.transpose(perm)
			
			T = tensordot(M, m, axes=([2],[0]))
			
			return T, [(j,M)]
			
		if ext_n==2:
			#
			# Two external legs.
			#
			
			k0,j0,m0 = in_msg_list[0]
			k1,j1,m1 = in_msg_list[1]
			
			
			#
			# Move the external legs to the end
			#
			
			perm = (v0_loc,v2_loc,k0,k1)
			M = M.transpose(perm)
			
			#
			# Now M legs are: [v0,v2,k0,k1]
			#
			
			M0 = tensordot(M, m1, axes=([3],[0]))
			M1 = tensordot(M, m0, axes=([2],[0]))
			T  = tensordot(M0, m0, axes=([2],[0]))
			
			return T, [(j0,M0),(j1,M1)]
			


	#
	# ~~~~~~~~~~~~~~~~~~~~~~~~  insideout_loop_SL
	# 
	
	def insideout_loop_SL(vs):
		
		r"""
		
		Given a loop described by the vertices vs=[v0, v1, ...], calculate
		its outgoing messages m0, m1, ... by contracting the loop together 
		with its incoming messages.
		
		The algorithm uses a first pass where it calculates the contraction
		of the "transfer-matrix" from the left, and then does another
		pass from the right. This enables it to calculate the loop env
		at every point, from which the outer message is calculated. 
		
		A sketch of the bookkeeping is found in insideout_loop_SL-bookkeping.pdf
		
		Input Parameters:
		------------------
		
		vs --- The list of vertices that make up the loop
		
		Output:
		-------
		out_m_list --- a list [(i,j,m), ...] where m is the outgoing
		               i-->j message (i is part of the loop)
		               
		               The order of the messages is according to the order
		               of the vertices in vs, and the order of the external
		               legs in each v in vs.
		
		
		"""
		
		elog = False

		if elog:
			print(f"Entering insideout_loop_SL with vs={vs}:")

		l = len(vs)
		
		#
		# Create the list of contracted tensors along the loop. 
		# The contraction is for all the external messages, so that
		# each contracted tensor contains only 2 remaining indices.
		#
		contT_list = []
		for i in range(l):
			v0 = vs[(i-1) % l]
			v1 = vs[i]
			v2 = vs[(i+1) % l]
			
			T, partials = clegs(v0, v1, v2)
					
			contT_list.append((T, partials))
			
				
		
		#
		# Dimension of the leg connecting T_{l-1} --- T_0
		#
		
		T,partials = contT_list[0]
		d0=T.shape[0]
		
		#
		# Create the list of all left contractions
		#
		# (L_k)_{al_k, al_0} = (T_k)_{al_k,al_k+1} ... T_{l-1}_{al_{l-1},al_0}
		#
		ML = eye(d0)
		Lmat_list = [None]*(l+1)
		Lmat_list[l] = ML
		for i in range(l-1,-1,-1):
			T,partials = contT_list[i]
			ML = T@ML
			ML = ML/norm(ML)
			Lmat_list[i] = ML
			
		#
		# Now go over v0 -> v_{l-1} and contract from the right, and 
		# use the left contraction to get the loop env and then the
		# outgoing message
		#
		
		if elog:
			print("\n\n")
			print("Entering insideout_loop_SL main Loop:")
		
		out_m_list = []
		
		MR = eye(d0)
		
		for i in range(l):
			if elog:
				print(f"Calculating out-messages of {vs[i]}")
			
			#
			# MR = (R_k)_{al_0,al_k} = (T_0)_{al_0,al_1}...(T_{k-1})_{al_k-1,al_k}
			#
			# Therefore R_0 = Id, R_1 = T_0, ...
			#           
			#
			
			#
			# (env_k)_{al_{k+1},al_k} = L_{k+1} R_k
			#
			env = Lmat_list[i+1]@MR
			
			# Define R_{k+1}
			
			T,partials = contT_list[i]
			MR = MR@T
			MR = MR/norm(MR)
			
			# 
			# Now we go over the partials and contract the 3-legs tensors
			# there with the env tensor to create the outgoing messages
			# from that vertex
			#
			
			for (j,M) in partials:
				
				#
				# M has legs:   (v0, v2, out-leg) 
				# env has legs: (v2, v0) 
				#
				
				m = tensordot(env, M, axes=([0,1],[1,0]))
				
				v1 = vs[i]
				out_m_list.append( (v1,j,m) )

		if elog:
			print("\n")
			print(f"insideout_loop_SL Final out messages: ")
			for k,(i,j,m) in enumerate(out_m_list):
				print(f"m[{k}]: {i} --> {j}  d={m.shape}")
			
		return out_m_list
				
		
 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	elog = False
	
	#
	# First check if we are in a single-layer (SL) or double-layer (DL) 
	# mode. In the DL mode, the tensors have a physical leg, and the actual
	# BP message is calculated by contracting T and T^* along that 
	# physical leg. 
	#
	# NOTE: currently, only SL mode is supported.
	#
	
	mode='SL'  # Single-Layer mode
		
	if mode=='DL':
		print("cluster_qbp error: currently double layer (DL) mode is not "\
			"supported")
		exit(1)
		

	n = len(T_list)    # no. of tensors
	nc = len(c_list)   # no. of clusters

	#
	# If initial_m is given as a list of messages --- it is the list
	# of initial messages to use. Otherwise, it is a string 'R' or 'U', 
	# telling us what kind of initial messages to create.
	#

	if type(initial_m) is list:
		m_list = initial_m
	else:

		#
		# In such case we should create the initial messages. We first
		# initialize a 2D list of size n\times n which holds the i->j
		# message
		#
		
		m_list = [ [None]*n for i in range(n)]  
				
		#
		# Go over all tensors, and for each tensor i go over its legs, and 
		# find its neighbors j
		#
		for i in range(n):
			
			no_legs = len(e_list[i])
			
			
			for leg in range(no_legs):
				
				e = e_list[i][leg]
				j = adj_vert(i, e, e_dict)
						
				#
				# Now assign a normalized random message. When in a ket mode, 
				# the message is a PSD [D,D] matrix. Otherwise it is a positive 
				# D vector.
				#
				
				#
				# We are on Single-Layer mode --- our messages are vectors
				#
				
				if type(T_list[i]) is np.ndarray:
					D = T_list[i].shape[leg]
				else:
					D = 2
				
				if initial_m =='R':
					message = np.random.uniform(size=[D])
				else:
					message = ones(D)
					
				message = message/sum(message)
					
				m_list[i][j] = message
				
	
	#
	# ===================== Main Iteration Loop =========================
	#
	
	err = 1.0
	iter_no = 0
	
	if elog:
		
		if mode=='SL':
			mode_s='Single-Layer'
		else:
			mode_s = 'Double-Layer'
		
		print("\n\n")
		print(f"Entering main cluster-qbp loop in {mode_s} mode")
		print("-----------------------------------------------------\n")
		


	
	while err>delta and iter_no < max_iter:
		
		iter_no += 1

		err = 0.0
		err_n = 0
		
		if elog:
			print(f"----------   Entering cluster_qbp round {iter_no}   -----------")
		
		
		#
		# Determine the order of the clusters in which the insideout is 
		# called. 
		#
		if type(corder)==str:
			if corder=='sequential':
				clusters_order = range(nc)
			elif corder=='random':
				clusters_order = np.random.permutation(nc)
			else:
				print(f"cluster_qbp error: illegal corder (given corder='{corder}')")
				exit(1)
		else:
			clusters_order = corder

		#
		# ---------------------------------------------------------------
		#                           Main Loop:
		#
		# Go over all clusters c in clusters_order and for each cluster 
		# calculate its outgoing messages based on its incoming ones.
		#
		# ---------------------------------------------------------------
		#
		
		for c in clusters_order:

			ctype, vs, params = c_list[c]

			if elog:
				print(f"Iteration {iter_no}: Entering cluster {ctype} with vs={vs}")


			
			#
			# Now see what type of a cluster we have. For each type, we run
			# the appropriate insideout function
			#
			
			if ctype in ['star', 'wcopy', 'xor']:
				# 
				# Create a list of incoming messages, send it to insideout_SL
				# and get a list of outgoing messages
				#
				i = vs[0]
				legs_no = len(e_list[i])
				
				
				in_m_list = []
				out_m_list = []
				
				for l in range(legs_no):
					e = e_list[i][l]
					
					j = adj_vert(i, e, e_dict)
					in_m_list.append(m_list[j][i])

				if ctype=='star':
					T = T_list[i]
					out_m1 = insideout_SL(T, in_m_list)
					
				elif ctype=='wcopy':
					out_m1 = insideout_wcopy(in_m_list, params)
					
				elif ctype=='xor':
					out_m1 = insideout_xor(in_m_list, params)
				
				
				out_m_list = []
				for l in range(legs_no):
					e = e_list[i][l]
					
					i1,i1_leg, j1, j1_leg = e_dict[e]
					
					j = (i1 if i1 !=i else j1)
					out_m_list.append( (i,j, out_m1[l]) )
					

			elif ctype=='loop':
				out_m_list = insideout_loop_SL(vs)

			
			#
			# At this point we have the list of outgoing messages from the 
			# cluster. Every item in the list is a tuple (i,j,m)
			# indicating the message m between i-->j
			#
			
			#
			# Normalize the messages and update the main list of messages
			#
			for i,j,out_m in out_m_list:

		
				if mode=='DL':
					#
					# Normalize according to the trace norm
					#
					
					print("!!! TODO: DL mode not yet implemented in cluster_qbp!")
					exit(0)
					
				else:
					#
					# Normalize in the L_1 norm (as if we're dealing with probs)
					#
					nr = norm(out_m, ord=1)
					out_m = out_m/nr
					tr = sum(out_m)
					tr_phase = tr/abs(tr)
					out_m = out_m/tr_phase
								

				#
				# Calculate the L_1 normalized error (both in DL and SL modes)
				#
				if mode=='DL':
					print("!!! TODO: DL mode not yet implemented in cluster_qbp!")
					exit(0)
				else:
					err += norm(m_list[i][j] - out_m, ord=1) 
					
				err_n += 1
					
				m_list[i][j] = (1-damping)*out_m + damping*m_list[i][j]

		
		# 
		# The error is the average L_1 distance divided by the total number
		# of coordinates if we stack all messages as one huge vector
		#
		
		if err_n>0:
			err = err/err_n
		
		if elog:
			print()
			print(f"cluster qbp iteration {iter_no}: BP-err = {err:.6g}\n")
		
	

	return m_list, err, iter_no
	
	



#
# ----------------------   insideout_wcopy   ---------------------------
#
def insideout_wcopy(in_m_list, p=None):
	r"""
	
	Provides the insideout functionality for a wcopy tensor. A wcopy
	tensor is a weighted copy tensor:
	
	              / p0   when   i1=i2=...=ik = 0
	W_{i1...ik} = 
	              \ p1   when   i1=i2=...=ik = 1
	              
	When the external parameter p is given then p0 = 1-p and p1=p. 
	When p=None then p0=p1=1, and we get the regular copy tensor.
	
	Input Parameters:
	-----------------
	
	in_m_list --- An incoming list of messages. Each message is assumed
	              to be a dim 2 vector (working with bits)
	              
	p         --- The wcopy weight parameter. When given, then p0=1-p, 
	              and p1=p. Otherwise, p0=p1=None.
	
	Output:
	-------
	A corresponding list of out BP messages
	
	
	"""
	
	ZERO_THRESH = 1e-15
	
	k = len(in_m_list)
	
	if p is None:
		m0, m1 = 1.0, 1.0
	else:
		m0, m1 = 1-p, p
	
	
	#
	# We now calculate the products 
	# m0 := m_1[0]*m_2[0]* ...
	# m1 := m_1[1]*m_2[1]* ...
	#
	# Then the outgoing message at leg i would be [m0/m_i[0], m1/m_i[1]]
	#
	# But we need to deal with "division-by-zero" with care. So there are
	# 3 cases when calculating m0 or m1:
	#
	# 1) All m_i[al] != 0, and therefore m{al} != 0
	# 2) There exactly one m_i[al] = 0. In such case, only the outgoing
	#    m_i[al] message is non-zero
	# 3) There are two or more m_i[al] = 0, and in such case all outgoing
	#    messages are zero for the al coordinate.
	#
	# So we first check which one of these options is valid. For al=0,1
	# we use two variables. For the al=0 we have:
	# m0zeros --- If there's one zero message in the 0 coordinate, then 
	#             this is its index.
	# m0alive --- If there are two or more zeros in the 0 cooridnate, 
	#             then m0alive=False. 
	#
	# The same is done for al=1
	#
	
	
	m0zeros, m1zeros = None, None
	m0alive, m1alive = True, True
	
	for i in range(k):
		
		if m0alive:
			if abs(in_m_list[i][0])>ZERO_THRESH:
				m0 *= in_m_list[i][0]
			else:
				if m0zeros is None:
					m0zeros = i
				else:
					m0alive = False

		if m1alive:
			if abs(in_m_list[i][1])>ZERO_THRESH:
				m1 *= in_m_list[i][1]
			else:
				if m1zeros is None:
					m1zeros = i
				else:
					m1alive = False
	
	out_m_list = []
	
	
	#
	# Once we have the products m0, m1, and we know if there is 
	# are one or more zeros, we can calculate the outgoing messages
	#
	for i in range(k):
		
		if m0alive:
			if m0zeros is None:
				m0i = m0/in_m_list[i][0]
			else:
				if m0zeros==i:
					m0i = m0
				else:
					m0i = 0
		else:
			m0i = 0
			
		if m1alive:
			if m1zeros is None:
				m1i = m1/in_m_list[i][1]
			else:
				if m1zeros==i:
					m1i = m1
				else:
					m1i = 0
		else:
			m1i = 0
		
		m = array([m0i, m1i])
		
		out_m_list.append(m)
		
	return out_m_list
	


#
# ----------------------   insideout_xor   ---------------------------
#
def insideout_xor(in_m_list, parity=0):
	r"""
	Provides the insideout functionality for the xor tensor. A xor 
	tensor with parity 0 or 1 is is given by:
	
	                   / 1  i1+i2+...+ik = parity mod 2
	xor[i1, ..., ik] = 
	                   \ 0  otherwise
	
	
	To calculate it we first add a k+1 leg that carries the parity and
	ask that the total parity is 0. Then we use the identity that xor is 
	equal to the 2*copy tensor after we tensor each one of its legs with
	the non-scaled Hadamarad gate [[1,1], [1,-1]].
	
	Therefore, we first create a copy tensor on k+1 legs, and then 
	multiply the legs by the unscaled Hadamard and finaly divide by 2.
	
	Input Parameters:
	-----------------
	in_m_list --- An incoming list of messages. Each message is assumed
	              to be a dim 2 vector (working with bits)
	              
	parity    --- The parity of the xor constraint.
	
	Output:
	-------
	A corresponding list of out BP messages
	
	
	
	"""
	
	
	k = len(in_m_list)
	
	Hm_list = [nscHgate@m for m in in_m_list]
	
	#
	# Add the parity of the xor as an additional message
	#
	if parity==0:
		Hm_list.append(nscHgate@array([1,0]))
	else:
		Hm_list.append(nscHgate@array([0,1]))
		
	out_list = insideout_wcopy(Hm_list)
	
	Hout_list = [nscHgate@m/2 for m in out_list]
	
	Hout_list = Hout_list[0:k] # remove the fictional parity message
	
	return Hout_list
	



#
# ----------------------- insideout_DL --------------------------------
#

def insideout_DL(T, in_m_list):
	
	r"""
	
	A DOUBLE-LAYER inside-out
	
	Given a ket tensor T and a list of incoming messages, calculate the 
	outgoing BP messages
	
	Input Parameters:
	-------------------
	
	T         --- The ket tensors. Should be of the form [d, D0, D1, D2, ...]
	
	in_m_list --- The incoming messages. If T has k logical legs, then
	              there would be k incoming messages, ordered by the legs.
	              
	              Each incoming message is a PSD [D_i,D^*_i] matrix, where
	              D_i is the ket leg and D^*_i is the bra leg.
	              
	Output:
	--------
	
	The list of outgoing messages, ordered according to the legs of T.
	
	
	"""
		
	legs_no = len(T.shape)-1
	
	
	#
	# We first create a list of partial contractions of ket-T to the 
	# messages, and bra-T to the messages. The outgoing messages will
	# then be the result of cross-contractions of tensors between these
	# two lists.
	#
	
	ketT = T
	braT = conj(T)
	
	ket_Ts = [ketT]
	bra_Ts = [braT]
	
	#
	# Contract messages legs_no-1, legs_no-2, ..., 1 to T_ket  AND
	# 1,2, ..., legs_no-1 to Tbra
	#
	
	for leg in range(legs_no-1):
		
		ketT = tensordot(ketT, in_m_list[legs_no-1-leg], axes=([legs_no-leg],[0]))
		braT = tensordot(braT, in_m_list[leg], axes=([1],[1]))

		ket_Ts.append(ketT)
		bra_Ts.append(braT)
		

	#
	# An example of the final lists with legs_no=4:
	# ----------------------------------------------
	#
	# j' = a contracted leg j
	#
	# ket_Ts: {[0,1,2,3,4], [0,1,2,3,4'], [0,1,2,4',3'], [0,1,4',3',2']}
	#
	# bra_Ts: {[0,1,2,3,4], [0,2,3,4,1'], [0,3,4,1',2'], [0,4,1',2',3']}
	#
	# So to get out-message i contract ket_Ts[legs_no-i-1] <==> bra_Ts[i]
	#
	# bra: [0,1,2,3,4],   [0,2,3,4,1'],  [0,3,4,1',2'], [0,4,1',2',3']
	# 	        ||            ||             ||             ||
	# ket: [0,1,4',3',2'] [0,1,2,4',3'], [0,1,2,3,4'],  [0,1,2,3,4]
	#        (0,4,3,2)      (0,4,3,2)
	
	
	#
	# Once we have the ket_Ts list and the bra_Ts list, we cross-contract
	# them to create the outgoig message:
	#
	# Out-going-message[i] := cont( bra[i] + ket[legs-i] )
	#
	#
	out_m_list = []
	
	full_axes=list(range(legs_no+1))
	
	bra_axes = [0] + list(range(2, legs_no+1))
	
	invL = [0] + list(range(legs_no, 1, -1))
	
	for i in range(legs_no):
		
		ket_axes = invL[:(legs_no-i)] + list(range(1, i+1))
		
		out_m = tensordot(ket_Ts[legs_no-i-1], bra_Ts[i], axes=(ket_axes, bra_axes))
		
		#
		# Make the outgoing message explicitly hermitian
		#
		out_m = 0.5*(out_m + conj(out_m.T))
		
		out_m_list.append(out_m)
		
	return out_m_list
	



#
# ----------------------- insideout_SL --------------------------------
#

def insideout_SL(T, in_m_list, direction='A'):
	
	r"""
	
	A SINGLE-LAYER inside-out
	
	Takes a tensor T_{i_0,..., i_{k-1}} and the set of *incoming* messages
	m_{i_0}, ..., m_{i_{k-1}} to each one of its legs, and computes the 
	set of outgoing messages.
	
	The computation is done in a recursive way. There are two *internal*
	modes of operation to the routine:
	'D' (Descending) --- This gives the outgoing message i_0.
	                     It is done by contracting T with m_{i_{k-1}}, 
	                     and recursively calling it on the resultant 
	                     tensor with 'D' mode.
	                     
	'A' (Ascending) --- Here we compute all the messages. This is done 
											by first invoking the routine with 'D' on the
											tensor, thereby getting the i_0 outgoing message.
											Then, we contract the i_0 leg, get the tensor
											T_{i_1, ..., i_{k-1}} and invoke the routine 
											recursively on T with mode 'A'.
	
	Parameters:
	------------
	
	T         --- The tensor on which we act
	
	in_m_list --- List of messages. The i'th message is a vector whose 
	              dimension should corrspond to the dimension of the i'th
	              leg of T.
	              
	direction --- Either 'A' (Ascending) or 'D' (Descending).
	
	              NOTE: This is an *internal* parameter, used in the 
	                    recursion. It should not be set when the function
	                    is called from the upper-most level.
	
	
	Output:
	--------
	The list of outgoing messages.
	
	"""
		
	
	legs_no = len(T.shape)
	
	if legs_no==1:
		
		out_m_list = [T]
	
	
	elif direction=='D':
	#
	# direction='D': contract the last leg, and call recursively with 'D'
	#

		m = in_m_list[-1]
				
		T1 = tensordot(T,m, axes=([legs_no-1],[0]))
			
			
		out_m_list = insideout_SL(T1, in_m_list[:(legs_no-1)], 'D')
		
	else:
	#
	# direction='A': Calculate the outgoing message of i_0, and then 
	#                contract i_0 and call recuresively with 'A' to 
	#                obtain the messages of i_1, i_2, ...
	#

		out_m_list1 = insideout_SL(T, in_m_list, 'D')  
	
		m = in_m_list[0]
		T1 = tensordot(T,m, axes=([0],[0]))
		out_m_list2 = insideout_SL(T1, in_m_list[1:], 'A') 
		
		out_m_list = out_m_list1 + out_m_list2
		
		
	
	return out_m_list
	
	
	

#
# --------------------------- qbp --------------------------------
#

def qbp(T_list, e_list, e_dict=None, initial_m='U', max_iter=10000, \
	delta=1e-6, damping=0.2, vorder='sequential'):

	r"""
	
	Runs a simple BP (belief Propagation) on a (closed) tensor network and 
	obtains the converged messages. 
	
	The TN can either be a quantum (<bra|ket> TN), in which case every 
	tensor in T is given as T[d, D0, D1, ...] or it can be classical,
	in which case every tensor is given as T[D0, D1, ...]
	
	In the quantum case the messages are PSD matrices of shape [D,D]. In 
	the classical case, they are vectors [D].
	
	The quantum/classical case is decided automatically by looking at 
	the shape of the tensor T_list[0] --- to see if it has a physical leg
	or not.
	
	(*) In the classical case, the messages are normalized so that sum(m)=1
	(*) In the quantum, we normalize so that Tr(m)=1 
	
	
	Parameters:
	-----------
	
	T_list --- The list of tensors that make up the TN (like in ncon)
	
	e_list ---	The labels of the edges (legs) of each tensor. This
									is a list of lists. For each tensor T with k legs
									there is a list [e_0, e_1, ..., e_{k-1}] of the labels
									of the legs. The same label must appear in the other
									tensor with which the leg is contracted.
	               
									Each label must be a positive integer.
									
	e_dict   --- A complementary structure to e_list. It is a dictionary
	             where the keys are the edges labels, and the value
	             is a 4-taple (i,i_leg, j, j_leg) with i<j.
	             
	             It is completely derivable from e_list. If omitted, 
	             it is calculated at the begining of the function.
									
	               
	initial_m   --- An optional list of initial messages. It can also be 
	                a string 'U' meaning unifrom initialization (1/D 
	                vector in the classical case or Id/D matrix in the 
	                quantum case), or it can be 'R', in which case we use
	                random normalized vector vector in the classical case
	                and a random PSD with Tr=1 in the quantum case).
	
									
	max_iter ---		A upper bound for the maxmial number of iterations
	
	delta			---			The distance between two consequtive sets of messages
									after which the iteration stops (the messages are 
									considered converged)
									
	damping --- 		A possible damping parameter to facilitate convergence.
									The new message is obtained by:
									
									(1-damping)*new_m + damping*old_m
									
									
	vorder ---      The order of the vertices in which BP is run; on each
	                round the algorithm go over all vertices according to
	                that order, and runs the insideout function. Takes all
	                the incoming messages to that vertex and calculate 
	                the outgoing.
	                
	                There are 3 possibilities:
	                'sequential' --- go over v=0,1,2,3,...
	                'random'     --- picks a random order
	                actual list  --- in  this case the list contains the
	                                 order of the verices.
	                  
	
	Output:
	-------
	
	The list of converged messages. This is a double list m_list[i][j]
	which holds the i->j message.
	
	In the double layer (DL) case, the each message is a matrix [D,D]. 
	Otherwise, it is a vector [D]

	
	
	
	"""
	
	TR_NORM_EPS = 1e-50

	elog = False
	
	#
	# First check if we are classical (ketbra) or quantum (ket).
	# In the quantum leg the tensors have a physical leg, and the actual
	# BP message is calculated by contracting T and T^* along that 
	# physical leg. 
	#
	
	if len(T_list[0].shape)==len(e_list[0]):
		mode='SL'  # Single-Layer mode
	else:
		mode='DL'  # Double-Layer mode (there's an extra phyiscal leg)
		
	
	#
	# If e_dict is not given, then calculate it
	#
	
	if e_dict is None:
		e_dict = calc_e_dict(e_list)

	n = len(T_list)

	#
	# If initial_m is given as a list of messages --- it is the list
	# of initial messages to use. Otherwise, it is a string 'R' or 'U', 
	# telling us what kind of initial messages to create.
	#

	if type(initial_m) is list:
		m_list = initial_m
	else:

		#
		# In such case we should create the initial messages. We first
		# initialize a 2D list of size n\times n which holds the i->j
		# message
		#
		
		m_list = [ [None]*n for i in range(n)]  
				
		#
		# Go over all tensors, and for each tensor i go over its legs, and 
		# find its neighbors j
		#
		for i in range(n):
			
			no_legs = len(e_list[i])
			
			for leg in range(no_legs):
					
				e = e_list[i][leg]
				
				i1,i1_leg, j1,j1_leg=e_dict[e]
				
				
				j = (i1 if i1 !=i else j1)
						
				#
				# Now assign a normalized random message. When in a ket mode, 
				# the message is a PSD [D,D] matrix. Otherwise it is a positive 
				# D vector.
				#
				
				if mode=='DL':
					#
					# We are on double-layer mode --- our messages are PSD matrices
					#
					D = T_list[i].shape[leg+1]
					
					if initial_m=='R':
						#
						# Use random PSD
						#
						ms = np.random.normal(size=[D,D])
						message = ms@ms.T
						message = message/trace(message)
					else:
						message = eye(D)/D
						
				else:
					#
					# We are on Single-Layer mode --- our messages are vectors
					#
					
					D = T_list[i].shape[leg]
					
					if initial_m =='R':
						message = np.random.uniform(size=[D])
					else:
						message = ones(D)
						
					message = message/sum(message)
					
				m_list[i][j] = message
				
	
	#
	# ===================== Main Iteration Loop =========================
	#
	
	err = 1.0
	iter_no = 0
	
	if elog:
		
		if mode=='SL':
			mode_s='Single-Layer'
		else:
			mode_s = 'Double-Layer'
		
		print("\n\n")
		print(f"Entering main qbp loop in {mode_s} mode")
		print("--------------------------------------------\n")

	while err>delta and iter_no < max_iter:
		
		iter_no += 1

		err = 0.0
		err_n = 0
		
		#
		# ---------------------------------------------------------------
		# Go over all vertices, and for each vertex find its outgoing 
		# messages from its incoming messages.
		# ---------------------------------------------------------------
		#
		
		#
		# Determine the order of the vertices in which the insideout is 
		# called. 
		#
		if type(vorder)==str:
			if vorder=='sequential':
				vertices_order = range(n)
			elif vorder=='random':
				vertices_order = np.random.permutation(n)
			else:
				print(f"qbp error: illegal vorder (given vorder='{vorder}')")
				exit(1)
		else:
			vertices_order = vorder
			
		for i in vertices_order:
				
			T = T_list[i]
			legs_no = len(e_list[i])
						
			# 
			# create a list of incoming messages
			#
			in_m_list = []   
			for l in range(legs_no):
				e = e_list[i][l]
				
				i1,i1_leg, j1, j1_leg = e_dict[e]
				
				j = (i1 if i1 !=i else j1)
				in_m_list.append(m_list[j][i])
				
				
			#
			# Calculate the outgoing messages using the insideout routine
			# (either in DL mode or in the SL mode)
			#
			if mode=='DL':
				out_m_list = insideout_DL(T, in_m_list)
			else:
				out_m_list = insideout_SL(T, in_m_list)
			
			#
			# Normalize the messages and update the main list of messages
			#
			for l in range(legs_no):

		
				if mode=='DL':
					#
					# Normalize according to the trace norm and make sure
					# that the trace of matrix message is a positive real.
					#
					nr = norm(out_m_list[l], ord='nuc')
					out_m_list[l] = out_m_list[l]/nr
					tr = trace(out_m_list[l])
					if abs(tr)>TR_NORM_EPS:
						tr_phase = tr/abs(tr)
						out_m_list[l] = out_m_list[l]/tr_phase
				else:
					#
					# Normalize in the L_1 norm (as if we're dealing with probs)
					# and make sure that the sum of each message is a positive
					# real.
					#
					nr = norm(out_m_list[l], ord=1)
					out_m_list[l] = out_m_list[l]/nr
					tr = sum(out_m_list[l])
					if abs(tr)>TR_NORM_EPS:
						tr_phase = tr/abs(tr)
						out_m_list[l] = out_m_list[l]/tr_phase
								
					#
				# Find the vertex j to which the message goes, and 
				# then find the old i->j message.
				#
				
				e = e_list[i][l]
				
				i1,i1_leg, j1,j1_leg = e_dict[e]
				j = (i1 if i1 !=i else j1)

				#
				# Calculate the L_1 normalized error (both in DL and SL modes)
				#
				if mode=='DL':
					err += norm(m_list[i][j] - out_m_list[l], ord='nuc') 
				else:
					err += norm(m_list[i][j] - out_m_list[l], ord=1) 
					
				err_n += 1
					
				m_list[i][j] = (1-damping)*out_m_list[l] + damping*m_list[i][j]

		
		# 
		# The error is the average L_1 distance divided by the total number
		# of coordinates if we stack all messages as one huge vector
		#
		
		if err_n>0:
			err = err/err_n
		
		if elog:
			print(f"qbp iter {iter_no}: BP-err = {err:.6g}")
		
	

	return m_list, err, iter_no
	
	
	
