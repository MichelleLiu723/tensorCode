
import sys
import pylab as plt
import pickle
from core_code import print_ranks
import numpy as np

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.style.use('ggplot')
from matplotlib import rc
rc('text', usetex=True)

fig_args={'figsize':(10,3)}
subplot_counter = 1

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f"usage: python {sys.argv[0]} [results-pickle-file(s)]\n \t results-pickle-file(s): pickle file(s) containing results")

	plt.figure(1,**fig_args)
	plt.figure(2,**fig_args)
	for fn in sys.argv[1:]:
		with open(fn,'rb') as f:
			results = pickle.load(f)
		results["Greedy"] = results["greedy"]
		del results["greedy"]
		results["Random walk"] = results["randomwalk"]
		del results["randomwalk"]
		methods = ["Greedy", "CP", "Tucker", "TT","Random walk"]

		plt.figure(1)
		plt.subplot(1,3,subplot_counter)
		for method in methods:
			dloss = [r['loss'] for r in results[method]]
			num_params = [r['num_params'] for r in results[method]]
			plt.plot(num_params,dloss,'.-')
		print(print_ranks(results["Greedy"][-1]["network"]))

		plt.axvline(x=results["_xp-infos_"]["targt_TN_params"],ls='--',c='black',lw=0.8)
		plt.legend(methods)# + ["opt. params"])
		plt.xlabel('parameters')
		plt.ylabel('reconstruction error')		
		plt.tight_layout()
		
		plt.figure(2)
		plt.subplot(1,3,subplot_counter)
		losses = moving_average([loss for r in results['Greedy'][1:] for loss in r['train_loss_hist']],n=10)
		losses_rw = moving_average([loss for r in results['Random walk'][1:] for loss in r['train_loss_hist']],n=10)
		plt.plot(range(len(losses)), losses)
		plt.plot(range(len(losses_rw)), losses_rw)
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.yscale('log')
		plt.legend(["Greedy","Random Walk"])# + ["opt. params"])
		#plt.title(fn)

		if "tt_" in fn:
			plt.title("TT target tensor")
		if "tr_" in fn:
			plt.title("TR target tensor")
		if "tri_" in fn:
			plt.title("Triangle target tensor")

		plt.tight_layout()
		subplot_counter += 1
	plt.show()