import matplotlib.pyplot as plt
import numpy as np
import APG, compare01,compare02,LS,HALS,getdata
import pandas as pd
  
def compareByRank():
	times = 300
	r = 100
	df = pd.DataFrame(columns=['Algorithm', 'r', 'Time (seconds)', 'Error'])
	for r in [25, 50, 75, 100]:
		V,W,H = getdata.getMatrix(r)
		
		_, _, _, HALS_Log = HALS.HALS_RRI(V, W, H, r, times)
		for i in range(len(HALS_Log)):
			df = df.append({'Algorithm': 'HALS_RRI', 'r': r, 'Time (seconds)': HALS_Log[1][i], 'Error': HALS_Log[0][i]}, ignore_index=True)
		
		_, _, _, LS_Log = LS.lee_seung_algorithm(V, W, H, times)
		for i in range(len(LS_Log)):
			df = df.append({'Algorithm': 'Lee-Seung', 'r': r, 'Time (seconds)':  LS_Log[1][i], 'Error': HALS_Log[0][i]}, ignore_index=True)
		_, _, _, APG_Log = APG.alternating_projected_gradient(V, W, H, times)
		for i in range(len(APG_Log)):
			df = df.append({'Algorithm': 'APG', 'r': r, 'Time (seconds)': APG_Log[1][i], 'Error': APG_Log[0][i]}, ignore_index=True)