import sys
import pylab as plt
import pickle
import numpy as np
import scipy.io
import core_code as cc
import torch
import random
np.random.seed(0)
torch.manual_seed(0)
random.seed(2)
from xp_completion_einstein import extract_observed_entries
#plt.style.use('ggplot')
from matplotlib import rc 
#rc('font',**{'sans-serif':['Helvetica']})

#rc('text', usetex=True)

def tt_parameters(imsize, rank):
    """
    This function returns the number of parameters of a TT decomposition with
    uniform rank [rank] of a tensor of shape imsize
    """
    rank = [rank] * (len(imsize)-1)
    d_left = np.cumprod(imsize)[:-1]
    d_right = np.cumprod(imsize[::-1])[::-1][1:]
    for i in range(len(rank)):
        rank[i] = np.min([rank[i],d_left[i],d_right[i]])
    rank.insert(0,1); rank.append(1)
    imsize.insert(0,1); imsize.append(1)
    L = [rank[i-1]*rank[i]*imsize[i] for i in range(1,len(imsize)-1)]
    return np.sum(L)


order='F'

nploty=7
nplotx=5
fig_args={'dpi':250,'figsize':(nplotx,nploty)}
fontsize=5
lw=2
ms=3
import core_code as cc

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f"usage: python {sys.argv[0]} [results-pickle-file(s)]\n \t results-pickle-file(s): pickle file(s) containing results")

	if 'einstein' in sys.argv[1]:
		xp = 'einstein'
		image = scipy.io.loadmat('data/Einstein.mat')['Data'] 
		image = np.asfortranarray(image).astype(float)
		im_size=image.shape
		image_reshaped = image.reshape([6,10,10,6,10,10,3],order=order)
		indices,values,im_missing = extract_observed_entries(image_reshaped,missing_rate=0.9,is_color_image=False)
		# results from Wang's paper
		tr_als_errors=[33.97,14.03,10.83,14.55]
		tr_als_params=[55*2**2,55*10**2,55*18**2,55*28**2]
	    
		tt_als_errors=[38.51, 22.89, 20.70, 23.19]
		tt_als_params=[202,3545,10089,22949]
	elif 'yale' in sys.argv[1]:
		xp = 'yale'
		tensor_shape = [6,8,6,7,8,8,19,2]
		image = scipy.io.loadmat('data/YaleBCrop025.mat')['I']
		image = np.asfortranarray(image).astype(float)
		im_size=image.shape
		image_reshaped = image.reshape(tensor_shape,order=order)
		indices,values,im_missing = extract_observed_entries(image_reshaped,missing_rate=0.9,is_color_image=False)
		tr_als_params = [np.sum(tensor_shape) * R**2 for R in [5,10,15,20,25,30]]
		print(tr_als_params)
		print(tensor_shape)
		tr_als_errors = [33.45, 24.67, 20.72, 18.47, 16.92, 16.25]

		tt_als_params = [1149, 3800, 7855, 13360, 20315, 28720]
		tt_als_errors = [37.08, 29.65, 27.91, 26.84, 26.16, 25.55,]
	elif 'video' in sys.argv[1]:
		xp = 'video'
		tensor_shape=[5,2,5,2,    13,2,5,2,   3,  5,17]
		image = scipy.io.loadmat('data/VideoData.mat')['Data'] 
		image = np.asfortranarray(image).astype(float)
		im_size=image.shape
		image_reshaped = image.reshape(tensor_shape,order=order)
		indices,values,im_missing = extract_observed_entries(image_reshaped,missing_rate=0.9,is_color_image=False)
		tr_als_params = [np.sum(tensor_shape) * R**2 for R in [10,15,20,25,30]]
		tr_als_errors = [13.90, 10.12, 8.13, 6.88, 6.25]

		tt_als_params = [tt_parameters(tensor_shape,R) for R in [10,15,20,25,30]]
		tt_als_errors = [19.16, 14.83, 16.42, 16.86, 16.99]




	greedy_errors = []
	greedy_params = []


	subplot_index = 1
	if xp=='einstein': #plot original and missing
		fig = plt.figure(**fig_args)
		
		plt.subplot(nploty,nplotx,subplot_index)
		subplot_index += 1
		plt.imshow(image.reshape(im_size,order='F')/255)
		plt.axis('off')
		plt.title(f"target image",fontsize=fontsize)
		plt.subplot(nploty,nplotx,subplot_index)
		subplot_index += 1
		plt.imshow(im_missing.reshape(im_size,order=order).astype(np.uint8))
		plt.axis('off')
		plt.title(f"observed pixels",fontsize=fontsize)

	for fn in sys.argv[1:]:
		with open(fn,'rb') as f:
			results = pickle.load(f)


		# losses = [loss for r in results[1:] for loss in r['train_loss_hist']]
		# plt.plot(range(len(losses)), losses)
		# plt.xlabel('epoch')
		# plt.ylabel('loss')
		# plt.legend(['greedy'])
		# #plt.title(fn)
		# plt.tight_layout()


		for res in results[1:]:

			im = cc.wire_network(res['network'],give_dense=True).detach().numpy().reshape(im_size,order='F')
			params=res['num_params']
			error=np.linalg.norm((image-im).ravel())/np.linalg.norm(image.ravel())*100

			greedy_errors.append(error)
			greedy_params.append(params)
			if len(greedy_params) > 1 and greedy_params[-1] == greedy_params[-2]:
				del greedy_params[-2]
				del greedy_errors[-2]
				subplot_index -= 1
			if xp == 'einstein':
				plt.subplot(nploty,nplotx,subplot_index)
				subplot_index += 1
				plt.imshow(im/255)
				plt.axis('off')
				plt.title(f"Iter. {subplot_index-3} - {params} param.\ntest error = {error:.2f}\\%",fontsize=fontsize)
				#f"Greedy-TL (iter={res_counter})\n{params} param."
			print(params,error)
			#plt.show()
		
	plt.tight_layout()
	plt.style.use('ggplot')
	plt.figure()
	plt.plot(greedy_params,greedy_errors,'o-',lw=lw,ms=ms)
	plt.plot(tr_als_params,tr_als_errors,'o-',lw=lw,ms=ms)
	plt.plot(tt_als_params,tt_als_errors,'o-',lw=lw,ms=ms)
	plt.legend("Greedy TR-ALS TT-ALS".split())
	plt.xlabel("parameters")
	plt.ylabel("relative error")
	plt.title('Einstein Image Completion')
	plt.tight_layout()
	
	plt.show()