import pandas as pd
from read_file_location import GetMatchedDatafile

Path='../MatchFile.csv'

df = pd.read_csv(Path)
print(df.describe)



