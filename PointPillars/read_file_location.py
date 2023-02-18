import pandas as pd

def GetMatchedDatafile(Path):
    df=pd.read_csv(Path)
    df = df.sample(frac=1).reset_index(drop=True)
    return [df['lidar_files'].tolist(),df['label_files'].tolist()]