import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pickle
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

mat = loadmat('./LYan-SEDEP/data_pu/all_mat/N15_M01_F10_KI07_16.mat')
struct_data = mat['N15_M01_F10_KI07_16']
Y_data = struct_data['Y']
signals = Y_data[0,0]['Data'][:,6]
new_signals=[]
for row in signals[0]:
    for value in row:
        new_signals.append(value)
new_signals = np.array(new_signals)
num_sample_per = 2048
num_samples = 970
interval = int((len(new_signals)-num_sample_per)/(num_samples-1))
data = np.zeros((num_samples, num_sample_per))
for m in range(num_samples):
    data[m, :] = new_signals[(m * interval):(m * interval + num_sample_per)]
transformer2 = QuantileTransformer(output_distribution='uniform')
transformer2.fit(data)
data = transformer2.transform(data)
x_train, x_test= train_test_split(data, test_size=0.3)
with open('./LYan-SEDEP/data_pu/x24.pkl', 'wb') as f:
    pickle.dump([x_train,x_test], f, pickle.HIGHEST_PROTOCOL)


