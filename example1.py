#!/usr/bin/python3
#----------------------------------------------------------
#
#                  example1.py
#             =======================
#
#  This program uses the qBP algorithm to contract 
#  a PEPS in the shape of the example given in the document describing
#  blockBP, in Sec.2.2
#
#  It creates a random PEPS with the topology of the TN in the example,
#  and calculates the <psi|psi> doubel-edge TN from it. Then it gives it
#  to the blockBP and uses the converged messages to calculate calculate
#  the expectation value <psi|Z_0|psi>.
#
#  It compares it to the expectation values from ncon. 
#
# History:
# ---------
#
# 8-Feb-2024  --- Initial version (copied from the blockBP example1.py)
#
# 29-May-2024 --- Removed TenQI,mpi4py dependence
#
#----------------------------------------------------------
#


import numpy as np

from numpy.linalg import norm

from numpy import zeros, ones, array, tensordot, sqrt, diag, dot, \
	reshape, transpose, conj, eye, trace, pi
	
from qbp import qbp

from ncon import ncon



#
# ------------------------- create_example  ---------------------------
#

def create_example_TN(d, D):
	
	"""
	 We create a random PEPS according to the example in Sec. 2.2 in the 
	 blockBP document, which has 12 spins.
	 
	 The physical legs are then contracted together in the <psi|psi> TN 
	 to create a double edge TN, on which we define the parameters that
	 are needed for blockBP contraction.
	 
	 Output:
	 ===========
	 1. The tensors of the original PEPS (single-edges)
	 
	 2. The blockBP parameters: 
	 
	   T_list, edges_list, pos_list, blocks_v_list, sedges_dict, 
	   sedges_list, blocks_con_list
	
	"""
	
	         
	D2 = D*D # Bond dimension of the double-edge model
	         
	
	
	#                T0        T1       T2       T3          T4
	edges_list = [ [1,2], [1,3,19,6], [5,3], [2,5,4,14], [19,4,7,8,20], 
	#   T5           T6        T7          T8               9
	[6,7,10], [8,10,9,11], [9,12], [11,20,15,18,17], [16,14,15],
	#  10            11
	[13,17,12], [16,18,13]]
	
	n = len(edges_list)
	
	# The list of the original PEPS tensors. First index in each tensor
	# is the physical leg.
	T_PEPS_list = []

	#
	# The list of the <psi|psi> TN tensors, which is sent to blockBP
	#
	T_list = []

	#
	# Create the local tensors as random tensors. Here we only use 
	# real tensors, for otherwise qbp will have problems converging
	# because of the small loops in the TN.
	#
	
	for eL in edges_list:
		k = len(eL)
		sh = [d] + [D]*k
		
		T = np.random.normal(size=sh) 
		T = T/norm(T)
		
		T_PEPS_list.append(T)
		
		#
		# Contract T with T^* and fuse the corresponding virtual legs 
		#
		
		perm=[]
		for i in range(k):
			perm = perm + [i,i+k]
	
		T2 = tensordot(T, conj(T), axes=([0],[0]))
		T2 = T2.transpose(perm)
		T2 = T2.reshape([D2]*k)
		
		T_list.append(T2)
		
		
	
	return T_PEPS_list, T_list, edges_list


#
#
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
#                             M  A  I  N
#
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
#


def main():
	
	random_seed = 3
	
	d = 2            # Physical bond dim
	D = 5            # Logical bond dim
	
	obs = array([[0.6,0.2+0.1j],[0.2-0.1j,0.1]])  # Observable we calculate
	
	ket = True       # Whether or not to use ket tensors
	
	#
	# Parameters for stopping condition and convergence
	#
	delta_err      = 1e-4  
	max_iterations = 100
	damping_rate   = 0.1
	
	#
	#  ~~~~~~~~~~~~~~~~~~~  Program starts here  ~~~~~~~~~~~~~~~~~~~~~~
	#
	


	np.random.seed(random_seed)
			
	#
	# Get the TN. The T_PEPS_list are the ket tensors (first leg is 
	# the physical leg of dim d). T_list are the ket-bra tensors
	#
	T_PEPS_list, T_list, edges_list = create_example_TN(d, D)

	T0 = T_PEPS_list[0]
	
	#	
	# ================================================================
	# (1) Use qBP to calculate the expectation value of obs at i=0
	# ================================================================
	#

	if ket:
		BP_tensors = T_PEPS_list
	else:
		BP_tensors = T_list

	m_list, err, iter_no = qbp(BP_tensors, edges_list,  \
			max_iter=max_iterations, delta=delta_err, \
			damping=damping_rate, use_ket=ket)
		
	print(f"... done with {iter_no} iterations and BP messages error={err}")

	print("\n\n\n")
	

	#
	# Once the qBP is over, we use the converged messages to calculate
	# <Z_0> --- the magnetization of qubit 0
	#
	# Note that vertex 0 has two neighboring vertices - 1,3.
	#
	
	if not ket:
		m_1_to_0 = m_list[1][0].reshape([D,D])
		m_3_to_0 = m_list[3][0].reshape([D,D])
	else:
		m_1_to_0 = m_list[1][0]
		m_3_to_0 = m_list[3][0]
	

	# T0 legs are [d,D1, D3]

	#
	# First, contract ket legs
	# 
	rho = tensordot(m_1_to_0, T0, axes=([0],[1])) 
	
	# now rho legs are [D1*, d, D3]
	
	rho = tensordot(rho, m_3_to_0, axes=([2],[0]))
	
	# now rho legs are [D1*, d, D3*]
	
	# normalize rho (so that Tr=1)
	
	rho = tensordot(rho, conj(T0), axes=([0,2], [1,2]))
	
	rho = rho/np.trace(rho)
	
	av_qBP = np.trace(rho@obs)

	#	
	# ================================================================
	# (2) Use ncon to calculate the expectation value of obs at i=0
	# ================================================================
	#
	
	D2 = D*D
	
	newT0 = tensordot(obs, T0, axes=([0],[0]))
	T = tensordot(newT0, conj(T0), axes=([0],[0]))
	T = T.transpose([0,2,1,3])
	T = T.reshape([D2,D2])
	
	#
	# calculate <psi|psi>
	#
	denominator = ncon(T_list, edges_list)
	
	#
	# Calculate <psi|obs|psi>
	#
	T_list[0] = T
	enumerator = ncon(T_list, edges_list)
	
	av_ncon = enumerator/denominator


	#	
	# ================================================================
	# (3) Print the results
	# ================================================================
	#

	print("Contraction Results:")
	print("---------------------\n")
	
	print(f"ncon: {av_ncon.real:.6g}    qBP: {av_qBP.real:.6g}")

	
main()
