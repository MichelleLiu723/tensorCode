import scipy.io
import pylab as plt
import random
import numpy as np
import core_code as cc
import torch


np.random.seed(0)
torch.manual_seed(0)
random.seed(2)

def extract_observed_entries(tensor, missing_rate, is_color_image=True, return_masked_tensor=True):
	shape = tensor.shape[:-1] if is_color_image else tensor.shape
	n_pixels = np.prod(shape)
	idx = random.sample(range(n_pixels),int((1-missing_rate)*n_pixels))
	obs_entries = np.unravel_index(idx,shape)
	obs_entries_indices = np.array(obs_entries)
	obs_entries_pixels = tensor[obs_entries]
	ret = [obs_entries_indices,obs_entries_pixels]
	if return_masked_tensor:
		masked_tensor = np.zeros(tensor.shape)
		masked_tensor[obs_entries] = tensor[obs_entries]
		ret.append(masked_tensor)
	return ret


import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
	filename = 'results-yale.pickle'
	tensor_shape = [6,8,6,7,8,8,19,2]
	order='F'
	image = scipy.io.loadmat('data/YaleBCrop025.mat')['I']
	image = np.asfortranarray(image).astype(float)
	im_size=image.shape
	image = image.reshape(tensor_shape,order=order)
	indices,values,im_missing = extract_observed_entries(image,missing_rate=0.9,is_color_image=False)




	dataset = (torch.Tensor(indices).to(torch.int64),torch.tensor(values))

	import sys
	initial_network=None
	if len(sys.argv)>1:
		with open(sys.argv[1],"rb") as f:
			results = pickle.load(f)
		initial_network = results[-1]['network']

	from greedy import greedy_completion
	results = greedy_completion(dataset,image.shape,initial_network=initial_network,filename=filename)

