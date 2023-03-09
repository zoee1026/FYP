import pandas as pd

def GetMatchedDatafile(Path):
    df=pd.read_csv(Path)
    train = df.sample(frac=0.9).reset_index(drop=True)
    test = df.drop(train.index)
    train.to_csv('train.csv')
    test.to_csv('test.csv')
    return [train['lidar_files'].tolist(),train['label_files'].tolist(),
            test['lidar_files'].tolist(),test['label_files'].tolist()
            ]


def TestModel(Path):
    df=pd.read_csv(Path)
    df = df.sample(frac=0.5).reset_index(drop=True)
    return [df['lidar_files'].tolist(),df['label_files'].tolist()]


