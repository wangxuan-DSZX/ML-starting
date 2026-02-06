# 数据处理：用于处理txt样本后，将数据归一化，再验证是否适用kNN算法
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 处理txt文档数据
def dating_progressing(filename):
    with open(filename) as f:
        lines_of_file= f.readlines()
    number_of_lines = len(lines_of_file)
    index = 0
    return_mat=np.zeros((number_of_lines, 3))
    class_labels=[]
    for line in lines_of_file:
        line=line.strip()
        list_from_line=line.split()
        return_mat[index,:]=list_from_line[0:3]
        class_labels.append(int(list_from_line[-1]))
        index +=1
    return return_mat,class_labels

# 数据归一化
def autonorm(return_mat):
    max_mat=np.max(return_mat,axis=0)
    min_mat=np.min(return_mat,axis=0)
    range_mat=max_mat-min_mat
    m=np.shape(return_mat)[0]
    norm_mat=np.zeros(np.shape(return_mat))
    norm_mat=return_mat-min_mat
    norm_mat=norm_mat/range_mat
    return norm_mat,range_mat,min_mat

# 检验三维数据有效性 更改sample.reshape即可
def kNNclassaltest(return_mat,class_labels,hoRatio):
    from sklearn.neighbors import KNeighborsClassifier
    kNN=KNeighborsClassifier(n_neighbors=2)
    norm_mat,range_mat,min_mat=autonorm(return_mat)
    X_train, X_test, y_train, y_test =train_test_split(norm_mat,class_labels,test_size=hoRatio,random_state=42)
    kNN.fit(X_train,y_train)
    m=np.shape(norm_mat)[0]
    num_class=int(m*hoRatio)
    error_class=0
    for i in range(num_class):
        sample=X_test[i].reshape(1,-1)
        classal_result=kNN.predict(sample)
        if classal_result!=y_test[i]:error_class+=1
    print(f'该测试样本错误率为：{error_class/num_class*100}%')

return_mat,class_labels=dating_progressing('dating.txt')
norm_mat,range_mat,min_mat=autonorm(return_mat)
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(return_mat[:,1],return_mat[:,2],s=15.0*np.array(class_labels),c=np.array(class_labels))
#plt.show()
# kNNclassaltest(return_mat,class_labels,0.2)



