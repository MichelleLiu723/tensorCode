
import sys
import pylab as plt
import pickle
plt.style.use('ggplot')

fig_args={'dpi':65,'figsize':(4,3)}

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f"usage: python {sys.argv[0]} [results-pickle-file]\n \t results-pickle-file: pickle file containing results")

	with open(sys.argv[1],'rb') as f:
		results = pickle.load(f)
	methods = list(results.keys())
	methods.remove("_xp-infos_")



	plt.figure(**fig_args)
	for method in methods:
		dloss = [r['loss'] for r in results[method]]
		num_params = [r['num_params'] for r in results[method]]
		plt.plot(num_params,dloss,'.-')

	plt.axvline(x=results["_xp-infos_"]["targt_TN_params"],ls='--',c='gray',lw=0.8)
	plt.legend(methods + ["opt. params"])
	plt.xlabel('number of parameters')
	plt.ylabel('loss')
	plt.title(sys.argv[1])
	plt.tight_layout()
	plt.figure(**fig_args)
	losses = [loss for r in results['greedy'][1:] for loss in r['train_loss_hist']]
	plt.plot(range(len(losses)), losses)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(['greedy'])
	plt.title(sys.argv[1])
	plt.tight_layout()
	plt.show()