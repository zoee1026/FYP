import numpy as np
from readers import Label3D
import json
import pandas as pd
from config import VehicaleClasses, Anchor_file





def ReadLabel(labelPath):
   with open(labelPath) as json_file:
        data = json.load(json_file)
        elements = []
        boundingBoxes = data['bounding_boxes']

        for box in boundingBoxes:
            print(box['center']['x'],box['center']['y'],box['center']['z']+box['height']/2,'--------------')

            element = Label3D(
                    str(box["object_id"]),
                    np.array([box['center']['x'],box['center']['y'],box['center']['z']+box['height']/2], dtype=np.float32),
                    np.array([box['width'],box['length'],box['height']], dtype=np.float32),
                    float(box['angle'])
                )
            # if element.classification =="dontcare":
            if element.classification not in list(VehicaleClasses.keys()):
                continue
            else:
                print (element)

if __name__ == '__main__':

    ReadLabel(r'C:\Users\Chan Kin Yan\Documents\GitHub\FYP\PointPillars\eval\eval_label\2020_12_02=09_06_42_832.bin.json')
    anchor_dims=np.round(np.array(pd.read_csv(Anchor_file,index_col=0).iloc[1:,:].values, dtype=np.float32).tolist(),3)
    # print('--------------------------------------------------------------------------------------')
    # print(anchor_dims)
    # print(anchor_dims.shape)

