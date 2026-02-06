import numpy as np
import matplotlib.pyplot as plt
from date_address import dating_progressing
from date_address import autonorm
from sklearn.neighbors import KNeighborsClassifier

def classify_person():
    class_mat, class_labels = dating_progressing('dating.txt')
    norm_mat,range_mat,min_mat = autonorm(class_mat)
    result_a = ['很感兴趣','较感兴趣','不感兴趣']
    gametimePercent=float(input('此人每月在游戏上的时间百分比:'))
    icecreamKilegram=float(input('此人每周消耗的冰激凌公升数：'))
    planetravelKilemeter=float(input('此人每年的飞行里程：'))
    inArray=np.array([planetravelKilemeter,gametimePercent,icecreamKilegram])
    inArray=(inArray-min_mat)/range_mat
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(norm_mat,class_labels)
    sample=inArray.reshape(1, -1)
    prediction = knn.predict(sample)
    print(f'你对这个人可能{result_a[prediction[0]-1]}')

classify_person()