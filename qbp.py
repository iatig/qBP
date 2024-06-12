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
#


import numpy as np

from numpy.linalg import norm
from numpy import sqrt, tensordot, array, zeros, ones, conj, trace



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

def qbp(T_list, edges_list, m_list=None, max_iter=10000, \
	delta=1e-6, damping=0.2, permute_order=False, use_ket=False):

	"""
	
	Runs a simple BP (belief Propagation) on a (closed) tensor network and 
	obtains the converged messages. If the TN is a <bra|ket> of PEPS, then
	the converged messages can be used to approximate the local 
	environment of the spins. This approximation is equal to the
	approximation of the simple update method.
	
	The TN is described in the notation of the ncon routine: a list of 
	tensors, together with a corresponding list of labels of edges. Here, 
	since we don't have any external edges, all labels must be positive 
	integers.
	
	The BP has two modes:
	1) If use_ket=True, then the tensors in T_list are of the form
	   T[d, D0, D1, ..].
	   
	   When calculating the BP messages, we contract T with T^* online.
	   
	   The BP messages in such case are density matrices. We normalize 
	   them so that they will have trace one.
	   
	2) if use_ket=False, we treat the problem as classical BP. The tensors
	   in T_list are of the form T[D0, D1, ...], and the messages are
	   vectors. We normalize them so that their *sum* is 1
	
	Parameters:
	-----------
	
	T_list --- The list of tensors that make up the TN (like in ncon)
	
	edges_list ---	The labels of the edges (legs) of each tensor. This
									is a list of lists. For each tensor T with k legs
									there is a list [e_0, e_1, ..., e_{k-1}] of the labels
									of the legs. The same label must appear in the other
									tensor with which the leg is contracted.
	               
									Each label must be a positive integer.
	               
	m_list   ---		An optional initial list of messages. This is a double 
									list. m_list[i][j] is the message i->j. This is a 
									vector whose dimension should match that of the leg
									that connects T_i with T_j.
									
									If m_list is not given, qbp starts with messages that
									are made from normalized random vectors with positive
									entries.
									
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
	                  
	use_ket     --- Whether or not the tensors are given as ket tensors.
	                In such case, the first leg (leg 0) is always the 
	                physical leg. The BP then contract the ket+bra 
	                online. In such case, the output messages are 
	                [D,D] matrices.
	                  
	
	Output:
	-------
	
	The list of converged messages. This is a double list m_list[i][j]
	which holds the i->j message.
	
	When use_ket=True, the each message is a matrix [D,D]. Otherwise, 
	It is a vector [D]

	
	
	
	"""

	log = True
	
	#
	# First, create a dictonary that tells the vertices of each edge
	# For positive (internal) edge, the value of the dictonary is (i,j), 
	# where i,j are the vertices connected by it. For negative edges
	# its (i,i).
	#
	
	vertices = {}
		
	n = len(T_list)
	
	for i in range(n):
		i_edges = edges_list[i]
		for e in i_edges:
						
			if e in vertices:
				(j1,j2) = vertices[e]
				vertices[e] = (i,j1)
			else:
				vertices[e] = (i,i)
				
		
	#
	# If m_list is empty, create an initial list of messages, where each
	# message is a normalized random vector with positive entries
	#
	
	if m_list is None:

		#
		# initialize a 2D list of size n\times n which holds the i->j
		# message
		#
		
		m_list = [ [None]*n for i in range(n)]  
		
		#
		# Go over all tensors, and for each tensor i go over its legs, and 
		# find its neighbors j
		#
		for i in range(n):
			
			no_legs = len(edges_list[i])
			
			for leg in range(no_legs):
				
					
				e = edges_list[i][leg]
				
				(i1,j1) = vertices[e]
				
				j = (i1 if i1 !=i else j1)
				
						
				#
				# Now assign a normalized random message. When in a ket mode, 
				# the message is a PSD [D,D] matrix. Otherwise it is a positive 
				# D vector.
				#
				
				if use_ket:
					#
					# Use random PSD
					#
					D = T_list[i].shape[leg+1]
					ms = np.random.normal(size=[D,D])
					message = ms@ms.T
					message = message/trace(message)
				else:
					D = T_list[i].shape[leg]
					message = ones(D)
					message = message/sum(message)
					
				m_list[i][j] = message
				
	
	#
	# ===================== Main Iteration Loop =========================
	#
	
	err = 1.0
	iter_no = 0
	
	if log:
		
		if use_ket:
			mode_s='ket'
		else:
			mode_s = 'ket-bra'
		
		print("\n\n")
		print(f"Entering main qbp loop (in {mode_s} mode)")
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
			legs_no = len(edges_list[i])
						
			# 
			# create a list of incoming messages
			#
			in_m_list = []   
			for l in range(legs_no):
				e = edges_list[i][l]
				(i1,j1) = vertices[e]
				j = (i1 if i1 !=i else j1)
				in_m_list.append(m_list[j][i])
				
				
			#
			# Calculate the outgoing messages using the insideout routine
			# (either in ket mode or in the usual ket-bra mode)
			#
			if use_ket:
				out_m_list = insideout_ket(T, in_m_list)
			else:
				out_m_list = insideout(T, in_m_list)
			
			#
			# Normalize the messages and update the main list of messages
			#
			for l in range(legs_no):


				if use_ket:
					out_m_list[l] = out_m_list[l]/trace(out_m_list[l])
				else:
					out_m_list[l] = out_m_list[l]/sum(out_m_list[l])
				
				#
				# Find the vertex j to which the message goes, and 
				# then find the old i->j message.
				#
				
				e = edges_list[i][l]
				(i1,j1) = vertices[e]
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
	
	
	
