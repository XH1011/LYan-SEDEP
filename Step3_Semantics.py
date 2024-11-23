import numpy as np
import pickle
import pandas as pd
from scipy.io import loadmat

def best_map(L1,L2,Matrix):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    #L1 真实标签 L2 预测
    Label1 = np.unique(L1) #保存数组L1的唯一元素，删除任何重复的值
    nClass1 = len(Label1) #Label1数据的数量
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i] #ind_cla1布尔数组，值为 True 的位置表示 L1 中与当前 Label1[i] 相等的元素位置
        ind_cla1 = ind_cla1.astype(float) #将ind_cla1转换为浮点型数组，将 True 转换为 1.0，False 转换为 0.0
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1) #?
    from munkres import Munkres
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index) #index中存储的是什么？
    c = index[:,1] #第二列的所有元素
    index_cluster = index[:, 0]+1 #聚类后类别顺序
    index_original = index[:, 1]+1  #聚类后原始类别随聚类顺序改变后的顺序
    print(index_cluster, index_original)
    newL2 = np.zeros(L2.shape)
    newMatrix = np.zeros(Matrix.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
        newMatrix[:,(index_original[i]-1)] = Matrix[:,(index_cluster[i]-1)]
    newMatrix_ind={}
    for i in range(nClass2):
        indices = np.where(newL2 == index_cluster[i])[0]
        newMatrix_per = newMatrix[indices, :]
        newMatrix_ind[index_cluster[i]] = []
        newMatrix_ind[index_cluster[i]].append(newMatrix_per)
    return newL2, newMatrix, newMatrix_ind, index_cluster,index_original
def smooth_probabilities(probs, epsilon=1e-4):
    return probs * (1 - epsilon) + (epsilon / probs.shape[1])

num_class=30
num_samples_per_class=291

# file_name = './LYan-SEDEP/DCAE/results_chopper/en_gen_x_test.pkl'
# file_name = './LYan-SEDEP/DCAE/results_pu/en_gen_x_test.pkl'
x_test = list(pickle.load(open(file_name, 'rb')))[0]
true_labels = pickle.load(open(file_name, 'rb'))[1]
true_labels = np.squeeze(true_labels)

from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=num_class,tol=1e-3,reg_covar=1e-6, max_iter=100, n_init=10, init_params='kmeans').fit(x_test)  # 指定聚类中心个数为4
pre_labels = gmm.predict(x_test)+1
probs = gmm.predict_proba(x_test)
mapped_pre_labels, mapped_probs, _, _, _ = best_map(true_labels, pre_labels, probs)
err_x = np.sum(true_labels[:] != mapped_pre_labels[:])  # 真实标签与聚类结果不一致的样本数量
acc = 1 - (err_x / (true_labels.shape[0]))
print('acc:', acc)
pre_labels_probs = np.argmax(probs, axis=1) + 1
mapped_pre_labels_probs, _, _, _, _ = best_map(true_labels, pre_labels_probs, probs)
err_x_probs = np.sum(true_labels[:] != mapped_pre_labels_probs[:])  # 真实标签与聚类结果不一致的样本数量
acc_probs = 1 - (err_x_probs / (true_labels.shape[0]))
print('acc_probs:', acc_probs)
cls_probs=[]
mean_cls_probs=[]
for i in range(num_class):
    cls_prob = mapped_probs[i*num_samples_per_class : (i+1)*num_samples_per_class]
    mean_cls_prob = np.mean(cls_prob, axis=0)
    cls_probs.append(cls_prob)
    mean_cls_probs.append(mean_cls_prob)
mean_cls_probs = np.array(mean_cls_probs)
data = {}
data['acc']=acc
for i in range(num_class):
    data[f'prob_mean_C{i + 1}'] = mean_cls_probs[:, i]
    df = pd.DataFrame(data)
    file_path = './Semantics/'
    file_name = f"Semantics_pu.xlsx"
    # file_name = f"Semantics_chopper.xlsx"
    file_full_path = file_path + file_name
    df.to_excel(file_full_path, index=False)


