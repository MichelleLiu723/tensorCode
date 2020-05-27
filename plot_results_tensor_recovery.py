
import sys
import pylab as plt
import pickle

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f"usage: python {sys.argv[0]} [results-pickle-file]\n \t results-pickle-file: pickle file containing results")

	with open('results-tt-cores-5.pickle','rb') as f:
		results = pickle.load(f)
	methods = results.keys()

	for method in methods:
	    dloss = [r['loss'] for r in results[method]]
	    num_params = [r['num_params'] for r in results[method]]
	    plt.plot(num_params,dloss)

	plt.legend(methods)
	plt.xlabel('number of parameters')
	plt.ylabel('loss')
	plt.figure()
	losses = [loss for r in results['greedy'][1:] for loss in r['train_loss_hist']]
	plt.plot(range(len(losses)), losses)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(['greedy'])
	plt.show()