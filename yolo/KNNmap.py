from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import joblib

def KNN():
    path='/media/sdb1/zoe/FYP/folder_root/LabelSummary.csv'
    df=pd.read_csv(path)
    knn = KNeighborsRegressor(n_neighbors=5,weights='distance')
    vector=df.sample(frac=.2, replace=True, random_state=1)

    X = vector[['x', 'y']]
    y = vector['angle']
    knn.fit(X, y)
    # joblib.dump(knn, "knn_model.joblib")
    return KNN