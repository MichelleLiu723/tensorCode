import scipy.io
import pylab as plt
import random
import numpy as np
import core_code as cc
import torch


np.random.seed(0)
torch.manual_seed(0)

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
	image = scipy.io.loadmat('data/Einstein.mat')['Data'] / 255
	indices,values,im_missing = extract_observed_entries(image,missing_rate=0.9,is_color_image=False)
	dataset = (torch.Tensor(indices).to(torch.int64),torch.tensor(values))

	goal_tn = torch.load('tt_cores_5.pt')
	data = cc.generate_completion_data(goal_tn,100)

	plt.figure()
	plt.imshow(im_missing)
	plt.ion()
	plt.show()
	plt.figure()

	from greedy import greedy_completion
	greedy_completion(dataset,image.shape)


