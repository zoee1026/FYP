from sklearn.neighbors import KNeighborsRegressor,KDTree
import pandas as pd
import pickle
import joblib
import numpy as np

def KNN():
    # path='/media/sdb1/zoe/FYP/folder_root/LabelSummary.csv'
    path=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\folder_root\ALLLabelSummary.csv'
    df=pd.read_csv(path)
    # knn = KNeighborsRegressor(n_neighbors=5,weights='distance')
    vector=df.sample(frac=.2, replace=True, random_state=1)

    X = vector[['x', 'y']].values
    y = vector['angle'].to_csv('knn_y_map.csv')
    kdt = KDTree(X)

    # knn.fit(X, y)

    filename = 'kdt.sav'
    pickle.dump(kdt, open(filename, 'wb'))

if __name__ == '__main__':
    KNN()
    kdt=pickle.load(open('kdt.sav', 'rb'))

    y=pd.read_csv('knn_y_map.csv')['angle'].values
    _, indices = kdt.query( [[-1.471259,10.611051]], k=5)
    y_pred = np.mean(y[indices], axis=1)[0]
    print(y_pred)
 