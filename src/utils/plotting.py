import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc

def plot_score(scores, labels, weights, num_bins=40, density=True, outname=None, save=False, show=False):
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

def plotROC(test_scores, test_labels, test_weights, train_scores=None, train_labels=None, train_weights=None, val_scores=None, val_labels=None, val_weights=None, outname=None, save=False, show=False):
    plt.grid(color='gray', linestyle='--', linewidth=1)
    
    if type(train_scores) == np.ndarray:
        fpr_train, tpr_train, _ = roc_curve(train_labels, train_scores, sample_weight=train_weights)
        tpr_train, fpr_train = np.array(list(zip(*sorted(zip(tpr_train, fpr_train)))))
        fnr_train = 1.0 - fpr_train
        roc_auc_train = 1 - auc(tpr_train, fpr_train)
        plt.plot(tpr_train, fnr_train, label='Train set, area = %0.6f' % roc_auc_train, color='black', linestyle='dotted')
    
    if type(val_scores) == np.ndarray:
        fpr_val, tpr_val, _ = roc_curve(val_labels, val_scores, sample_weight=val_weights)
        tpr_val, fpr_val = np.array(list(zip(*sorted(zip(tpr_val, fpr_val)))))
        fnr_val = 1.0 - fpr_val
        roc_auc_val = 1 - auc(tpr_val, fpr_val)
        plt.plot(tpr_val, fnr_val, label='Val set, area = %0.6f' % roc_auc_val, color='blue', linestyle='dashdot')
    
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_scores, sample_weight=test_weights)
    tpr_test, fpr_test = np.array(list(zip(*sorted(zip(tpr_test, fpr_test)))))
    fnr_test = 1.0 - fpr_test
    roc_auc_test = 1 - auc(tpr_test, fpr_test)
    plt.plot(tpr_test, fnr_test, label='Test set, area = %0.6f' % roc_auc_test, color='red', linestyle='dashed')   
        
    plt.plot([0, 1], [1, 0], linestyle='--', color='black', label='Luck')
    plt.xlabel('Signal acceptance')
    plt.ylabel('Background rejection')
    plt.title('Receiver operating characteristic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.legend(loc='lower left', framealpha=1.0)
    plt.tight_layout()

    if show:
        plt.show()
    if save:
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        plt.savefig(outname)
    plt.clf()
