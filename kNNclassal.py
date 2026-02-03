import numpy as np
from collections import Counter
import kNNexampledate
points,types=kNNexampledate.randomPoint()
def kNNclassal(points, types,k,measuredate):
    measuredate=np.array(measuredate)
    points=np.array(points)
    assert points.ndim==2
    assert points.shape
    d=len(points)
    distances = []
    i=0;j=0;
    for i in range(d):
        dist=np.sqrt(np.sum((points[i] - measuredate) ** 2))
        distances.append(dist)
    k_index = np.argsort(distances)[:k]
    k_types=[types[i] for i in k_index]
    real_type=Counter(k_types).most_common(1)[0][0]
    return real_type
real_type=kNNclassal(points, types,2,[10,10])
print(real_type)


