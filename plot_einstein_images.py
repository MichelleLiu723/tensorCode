import sys
import pylab as plt
import pickle
import numpy as np
import scipy.io
import core_code as cc
import torch
import random
from matplotlib import rc 
#rc('font',**{'sans-serif':['Helvetica']})

rc('text', usetex=True)
np.random.seed(0)
torch.manual_seed(0)
random.seed(2)
from xp_completion_einstein import extract_observed_entries
#plt.style.use('ggplot')

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

nploty=2
nplotx=9
subplot_index = 1
fontsize=6
fig_args={'dpi':250,'figsize':(nplotx-2,nploty)}
import core_code as cc

if __name__ == '__main__':

	image = scipy.io.loadmat('data/Einstein.mat')['Data'] 
	image = np.asfortranarray(image).astype(float)
	im_size=image.shape
	tensor_shape = [6,10,10,6,10,10,3]
	image_reshaped = image.reshape(tensor_shape,order=order)
	indices,values,im_missing = extract_observed_entries(image_reshaped,missing_rate=0.9,is_color_image=False)
	# results from Wang's paper
	tr_als_errors=[33.97,14.03,10.83,14.55]
	tr_als_params=[55*2**2,55*10**2,55*18**2,55*28**2]
    
	tt_als_errors=[38.51, 22.89, 20.70, 23.19]
	tt_als_params=[202,3545,10089,22949]

	fig = plt.figure(**fig_args)
	plt.subplot(nploty,nplotx,subplot_index)
	subplot_index += 1
	plt.imshow(image.reshape(im_size,order='F')/255)
	plt.axis('off')
	plt.title(f"Original\nimage",fontsize=fontsize)

	for alg,rk,fn in [(alg,rk,f"einstein-{alg}-{rk}.mat")  for rk in [2,10,18,26] for alg in "TT TR".split()]:
		print(alg,rk,fn)

		im = scipy.io.loadmat('../TensorRingCompletion/'+fn)['Data_Recover_TR'] 
		im = np.asfortranarray(im).astype(float)

		plt.subplot(nploty,nplotx,subplot_index)
		subplot_index += 1
		plt.imshow(im/255)
		plt.axis('off')
		params = 55*rk**2 if alg=='TR' else tt_parameters(tensor_shape,rk)
		if rk==18:
			plt.title(r"$\textbf{" + f"{alg} (rank={rk})" + "}$" + f"\n{params} param.",fontsize=fontsize)
		else:
			plt.title(f"{alg} (rank={rk})\n{params} param.",fontsize=fontsize)


	plt.subplot(nploty,nplotx,subplot_index)
	subplot_index += 1
	plt.imshow(im_missing.reshape(im_size,order=order).astype(np.uint8))
	plt.axis('off')
	plt.title(f"Observed\npixels",fontsize=fontsize)

	res_counter = 0
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
		prev_param = None
		for res in results[1:]:
			res_counter += 1
			im = cc.wire_network(res['network'],give_dense=True).detach().numpy().reshape(im_size,order='F')
			prev_param = params
			params=res['num_params']
			if prev_param and prev_param==params:
				print("yo")
				res_counter -=1 
			if not res_counter in [4,6,10,12,17,23,26,31]:
				continue
			error=np.linalg.norm((image-im).ravel())/np.linalg.norm(image.ravel())*100

			plt.subplot(nploty,nplotx,subplot_index)
			subplot_index += 1
			plt.imshow(im/255)
			plt.axis('off')
			plt.title(f"Greedy (iter={res_counter})\n{params} param.",fontsize=fontsize)
			if res_counter == 26:
				plt.title(r'$\textbf{Greedy (iter=' + f'{res_counter}'+')}$' + f"\n{params} param.",fontsize=fontsize)
	plt.show()
	sys.exit(0)


	greedy_errors = []
	greedy_params = []


		
		

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

			if xp == 'einstein':
				plt.subplot(nploty,nplotx,subplot_index)
				subplot_index += 1
				plt.imshow(im/255)
				plt.axis('off')

				plt.title(f"#param.:{params},\ntest error:{error:.2f}%",fontsize=4)
		
	plt.tight_layout()
	plt.style.use('ggplot')
	plt.figure()
	plt.plot(greedy_params,greedy_errors,'-')
	plt.plot(tr_als_params,tr_als_errors,'-')
	plt.plot(tt_als_params,tt_als_errors,'-')
	plt.legend("Greedy TR-ALS TT-ALS".split())
	plt.xlabel("parameters")
	plt.ylabel("relative error")
	plt.tight_layout()
	
	plt.show()