from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import pickle
import joblib
import numpy as np

def KNN():
    # path='/media/sdb1/zoe/FYP/folder_root/LabelSummary.csv'
    path=r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\folder_root\ALLLabelSummary.csv'
    df=pd.read_csv(path)
    knn = KNeighborsRegressor(n_neighbors=5,weights='distance')
    vector=df.sample(frac=.2, replace=True, random_state=1)

    X = vector[['x', 'y']].values
    y = vector['angle'].values
    knn.fit(X, y)

    filename = 'knn.sav'
    pickle.dump(knn, open(filename, 'wb'))

if __name__ == '__main__':
    KNN()
    knn=pickle.load(open('knn.sav', 'rb'))
    print(knn.predict(np.array([[-10,10]]))[0])
