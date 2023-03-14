import numpy as np
import math as m
import open3d
  
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def Manal_R(  phi = 0.0705718,  theta = -0.2612746,psi = -0.017035 ):
   
    R_manual = Rx(phi) * Ry(theta) * Rz(psi)
    return R_manual
   
if __name__ == '__main__':

    R_manual=Manal_R()

    pts = open3d.geometry.PointCloud()
    R = pts.get_rotation_matrix_from_xyz((0.0705718, -0.2612746,-0.017035))
    print(R_manual)
    v1 = np.array([[1],[1],[1]])
    print(R_manual*v1)
    print('-----------------------')
    print(R)
    pts.points=open3d.utility.Vector3dVector(np.array([[1,1,1]]))
    pts.rotate(R, center=False)

    print(np.array(pts.points) )

