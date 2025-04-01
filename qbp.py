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
#  25-Feb-2024: Itai  Fixed a bug in insideout_ket: removed a conj() 
#                     function on the BP message that was contracted to 
#                     the bra tensor.
#
#  2-Jun-2024: Itai 1. Make the outgoing messages in insideout_ket 
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


import numpy as np

from numpy.linalg import norm
from numpy import sqrt, dot, tensordot, array, zeros, ones, conj, trace,\
	eye, log, pi







#
# ---------------------------- calc_e_dict  ----------------------------
#

def calc_e_dict(e_list):
	"""
	
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
# ------------------  get_Bethe_free_energy   --------------------------
#
def get_Bethe_free_energy(m_list, T_list, e_list, e_dict):
	
	"""
	
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
	
	#
	# Check if we're in quantum or classical mode
	#
	
	if len(T_list[0].shape)==len(e_list[0]):
		mode = 'C'
	else:
		mode = 'Q'
	
	
	n = len(T_list)
	
	#
	# First, normalize the messages such that the inner product of 
	# the messages along an edge = 1
	#
	# Tr( m_{i\to j} \cdot m_{j\to i} ) = 1
	#
	# See Lemma III.1 in the blockBP paper, arXiv:2402.04834
	#
	nr_m_list = [[None]*n for i in range(n)]

	if mode=='C':
		
		#
		# We're on classical mode; messages are vectors
		#
		
		for e in e_dict.keys():
			
			i,i_leg, j,j_leg = e_dict[e]
			
			nr = sqrt(sum(m_list[i][j]*m_list[j][i]))
			
			nr_m_list[i][j] = m_list[i][j]/nr
			nr_m_list[j][i] = m_list[j][i]/nr
	else:
		
		#
		# We're on quantum mode; messages are matrices
		#
		
		for e in e_dict.keys():
			
			i,i_leg, j,j_leg = e_dict[e]

			nr = sqrt(trace( m_list[i][j]@(m_list[j][i]).T ))
						
			nr_m_list[i][j] = m_list[i][j]/nr
			nr_m_list[j][i] = m_list[j][i]/nr
			
	F_bethe = 0.0j
	
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
			
				M = tensordot(M, nr_m_list[j][i],axes=([0],[0]))
			
			M = M.astype(np.complex128)
			F_bethe = F_bethe - log(M)
			print(f"M={M}  log={log(M)}")
			
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
			
				M = tensordot(M, nr_m_list[j][i],axes=([1],[0]))
				
			M = dot(M.flatten(), conj(T_list[i].flatten()))
			
			#
			# At this point, M is the contraction of the ket with all the 
			# ket-legs of the incoming messages. We now contract this with 
			# the bra
			#
			
			F_bethe = F_bethe - log(M.astype(np.complex128))
	
	
	
	
	#
	# Make the imaginary part minimal (rememeber that 2*pi multiples do
	# not matter)
	#
	F_bethe_i = F_bethe.imag % (2*pi)
	if F_bethe_i>2*pi-F_bethe_i:
		F_bethe_i = F_bethe_i - 2*pi
	
	F_bethe = F_bethe.real + 1j*F_bethe_i
	
	#
	# If we're on the real case, drop the imaginary part
	#
	if abs(F_bethe.imag)<1e-10*abs(F_bethe):
		F_bethe = F_bethe.real
		
	return F_bethe




#
# ----------------------- insideout_ket --------------------------------
#

def insideout_ket(T, in_m_list):
	
	"""
	
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
# ----------------------- insideout --------------------------------
#

def insideout(T, in_m_list, direction='A'):
	
	"""
	
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
			
			
		out_m_list = insideout(T1, in_m_list[:(legs_no-1)], 'D')
		
	else:
	#
	# direction='A': Calculate the outgoing message of i_0, and then 
	#                contract i_0 and call recuresively with 'A' to 
	#                obtain the messages of i_1, i_2, ...
	#

		out_m_list1 = insideout(T, in_m_list, 'D')  
	
		m = in_m_list[0]
		T1 = tensordot(T,m, axes=([0],[0]))
		out_m_list2 = insideout(T1, in_m_list[1:], 'A') 
		
		out_m_list = out_m_list1 + out_m_list2
		
		
	
	return out_m_list
	
	
	

#
# --------------------------- qbp --------------------------------
#

def qbp(T_list, e_list, e_dict=None, initial_m='U', max_iter=10000, \
	delta=1e-6, damping=0.2, permute_order=False):

	"""
	
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
									
									
	permute_order --- If True, then the messages at each round are 
	                  updated in a random permutation of the vertices. 
	                  Otherwise, it is done sequentially.
	                  
	
	Output:
	-------
	
	The list of converged messages. This is a double list m_list[i][j]
	which holds the i->j message.
	
	In the quantum case, the each message is a matrix [D,D]. Otherwise, 
	It is a vector [D]

	
	
	
	"""

	log = False
	
	#
	# First check if we are classical (ketbra) or quantum (ket).
	# In the quantum leg the tensors have a physical leg, and the actual
	# BP message is calculated by contracting T and T^* along that 
	# physical leg. 
	#
	
	if len(T_list[0].shape)==len(e_list[0]):
		mode='C'  # Classical mode
	else:
		mode='Q'  # Quantum mode (there's an extra phyiscal leg)
		
	
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
				
				if mode=='Q':
					#
					# We are on quantum mode --- our messages are PSD matrices
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
					# We are on classical mode --- our messages are vectors
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
	
	if log:
		
		if mode=='C':
			mode_s='Classical'
		else:
			mode_s = 'Quantum'
		
		print("\n\n")
		print(f"Entering main qbp loop in {mode_s} mode")
		print("-----------------------------------------\n")

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
		# If permute_order = True, then the order in which we calculate
		# the messages is random (we go over a random ordering of the 
		# vertices)
		#
		if permute_order:
			vertices_order = np.random.permutation(n)
		else:
			vertices_order = range(n)
			
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
			# (either in ket mode or in the usual ket-bra mode)
			#
			if mode=='Q':
				out_m_list = insideout_ket(T, in_m_list)
			else:
				out_m_list = insideout(T, in_m_list)
			
			#
			# Normalize the messages and update the main list of messages
			#
			for l in range(legs_no):


				if mode=='Q':
					out_m_list[l] = out_m_list[l]/trace(out_m_list[l])
				else:
					out_m_list[l] = out_m_list[l]/sum(out_m_list[l])
				
				#
				# Find the vertex j to which the message goes, and 
				# then find the old i->j message.
				#
				
				e = e_list[i][l]
				
				i1,i1_leg, j1,j1_leg = e_dict[e]
				j = (i1 if i1 !=i else j1)

				#
				# Calculate the L_1 normalized error (suitable both in ket and
				# ketbra modes)
				#
				err += 2*norm(m_list[i][j] - out_m_list[l], ord=1) \
					/(norm(out_m_list[l], ord=1) + norm(m_list[i][j], ord=1))
				err_n += 1
				
				message = (1-damping)*out_m_list[l] + damping*m_list[i][j]
								
				m_list[i][j] = message
		
		# 
		# The error is the average L_1 distance divided by the total number
		# of coordinates if we stack all messages as one huge vector
		#
		
		err = err/err_n
		
		if log:
			print(f"qbp iter {iter_no}: BP-err = {err:.6g}")
		
	

	return m_list, err, iter_no
	
	
	
