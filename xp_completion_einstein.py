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
	filename = 'results-einstein.pickle'
	order='F'
	image = scipy.io.loadmat('data/Einstein.mat')['Data'] 
	image = np.asfortranarray(image).astype(float)
	im_size=image.shape
	image = image.reshape([6,10,10,6,10,10,3],order=order)
	indices,values,im_missing = extract_observed_entries(image,missing_rate=0.9,is_color_image=False)



	def plot_image(tensor_list):
		plt.imshow(cc.wire_network(cc.copy_network(tensor_list),give_dense=True).detach().numpy().reshape(im_size,order=order)/255)

	dataset = (torch.Tensor(indices).to(torch.int64),torch.tensor(values))

	goal_tn = torch.load('tt_cores_5.pt')
	data = cc.generate_completion_data(goal_tn,100)

	plt.figure()
	plt.imshow(im_missing.reshape(im_size,order=order).astype(np.uint8))
	plt.ion()
	plt.draw()
	plt.pause(1)
	plt.figure()

	import sys
	import pickle
	initial_network=None
	if len(sys.argv)>1:
		with open(sys.argv[1],"rb") as f:
			results = pickle.load(f)
		initial_network = results[-1]['network']

	from greedy import greedy_completion
	results = greedy_completion(dataset,image.shape,plot_image=plot_image,initial_network=initial_network,filename=filename)
