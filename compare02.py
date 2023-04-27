import APG,LS,HALS,getdata
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.decomposition import NMF


def compare(log1, log2, log3):
    plt.plot(log1[0], log1[1], label = 'LS')
    plt.plot(log2[0], log2[1], label = 'APG')
    plt.plot(log3[0], log3[1], label = 'HALS')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Frobenius error')
    plt.legend()

def compareByTime():
	times = 300
	r = 100
	V,W,H = getdata.getMatrix(r)
	_,_, _, LS_Log = LS.lee_seung_algorithm(V, W, H, times)

	_,_, _, HALS_Log = HALS.HALS_RRI(V, W, H, r,times)

	_,_, _, APG_Log = APG.alternating_projected_gradient(V, W, H, times)

	compare(LS_Log,APG_Log,HALS_Log)