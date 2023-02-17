import pandas as pd
from read_file_location import GetMatchedDatafile

Path='../MatchFileFeb16.csv'

df = pd.read_csv(Path)
print(df.describe)

GetMatchedDatafile(Path)


