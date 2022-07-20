import matplotlib.pyplot as plt
import numpy as np
import os

def plot_score(scores, labels, weights, num_bins=40, density=True, outname=None, save=None, show=None):
    sig_scores = scores[np.where(labels == 1)]
    bkg_scores = scores[np.where(labels == 0)]
    sig_weights = weights[np.where(labels == 1)]
    bkg_weights = weights[np.where(labels == 0)]
    plt.hist(sig_scores, weights=sig_weights, range=(min(0, sig_scores.min()), max(1, sig_scores.max())), density=density, bins=num_bins, label="Signal", alpha=0.5)
    plt.hist(bkg_scores, weights=bkg_weights, range=(min(0, bkg_scores.min()), max(1, bkg_scores.max())), density=density, bins=num_bins, label="Background", alpha=0.5)
    plt.xlabel('NN score')
    plt.ylabel('Fraction of events')
    plt.title(r'Score distribution')
    plt.legend(loc='upper left')#, framealpha=0)
    plt.tight_layout()
    if show:
    	plt.show()
    if save:
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        plt.savefig(outname)
    plt.clf()