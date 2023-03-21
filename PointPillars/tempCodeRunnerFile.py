# def ReadResultInOneFile(labelPath):
#     with open(labelPath, "r") as f:
#         lines = f.readlines()
#         elements = []
#         for line in lines:
#             box = line.split()
#             if box[0] not in list(OutPutVehecleClasees.values()):
#                 continue
#             elements.append(box)

#         print(elements)  
#         return np.array(elements)