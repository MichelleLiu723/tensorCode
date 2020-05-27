import tensorly as tl 
import tensorly.decomposition
import numpy as np

def l2_distance(T1,T2):
	return tl.norm(T1-T2)

def CP(target,rank):
	lambd, factors = tl.decomposition.parafac(target,rank)
	print([A.shape for A in factors])
	num_params = np.sum([A.size for A in factors])
	return tl.kruskal_to_tensor((lambd,factors)), num_params

def TT(target,rank):
	factors = tl.decomposition.matrix_product_state(target,rank)
	num_params = np.sum([A.size for A in factors])
	return tl.mps_to_tensor(factors), num_params

def Tucker(target,rank):
	ranks = [min(rank,d) for d in target.shape]
	(G, factors) = tl.decomposition.tucker(target,ranks)
	num_params = np.sum([G.size] + [A.size for A in factors])
	return tl.tucker_to_tensor((G,factors)), num_params

def incremental_tensor_recovery(target, decomposition, loss_threshold=1e-5, max_num_params=1500, verbose=False):
	if decomposition not in "CP TT Tucker".split():
		raise(NotImplementedError())

	results = []

	rank = 1
	loss,num_params = np.infty, 0
	decomposition_algo = {"TT":TT, "Tucker":Tucker, "CP":CP}
	it=0
	while (loss > loss_threshold) and (num_params < max_num_params):
		it += 1
		tensor,num_params = decomposition_algo[decomposition](target,rank)
		loss = l2_distance(target,tensor)
		results.append({"iter":it,"num_params":num_params,"loss":loss,"rank":rank})
		rank += 1
		if verbose:
			print(results[-1])

	return results


if __name__ == "__main__":
	import sys
	import core_code as cc
	import torch

	tl.set_backend('numpy')
	target_file = "tt_cores_5.pt"
	goal_tn = torch.load(target_file)
	target = cc.wire_network(goal_tn,give_dense=True).numpy()

	results = {}
	for decomp in "CP TT Tucker".split():
		print(f"running {decomp} decomposition...")
		results[decomp] = incremental_tensor_recovery(target,decomp,verbose=True)

