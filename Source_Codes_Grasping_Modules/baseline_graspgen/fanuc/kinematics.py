"""M-20iD/25 flange kinematics, with the FANUC J2/J3 coupling convention."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
# From FANUC-CORPORATION/fanuc_description/m20_25_18d_urdf_macro.xacro.
ORIGINS=[[0,0,.425],[.075,0,0],[0,0,.84],[0,0,.215],[.89,0,0],[0,0,0]]
AXES=[[0,0,1],[0,1,0],[0,-1,0],[-1,0,0],[0,-1,0],[-1,0,0]]
# Conservative controller ranges; native MoveTo must accept and read back every command.
LIMITS=np.deg2rad([[-185,185],[-100,160],[-90,90],[-200,200],[-140,140],[-270,270]])
def translation(xyz):
    T=np.eye(4);T[:3,3]=xyz;return T
def fk(controller_q):
    q=np.array(controller_q,dtype=float).copy();q[2]+=q[1]
    T=np.eye(4)
    for origin,axis,angle in zip(ORIGINS,AXES,q):
        R=np.eye(4);R[:3,:3]=Rotation.from_rotvec(np.array(axis)*angle).as_matrix()
        T=T@translation(origin)@R
    R=np.eye(4);R[:3,:3]=Rotation.from_euler('xyz',[180,-90,0],degrees=True).as_matrix()
    return T@translation([.09,0,0])@R
def from_wpr(p):
    T=np.eye(4);T[:3,3]=np.array(p[:3])/1000;T[:3,:3]=Rotation.from_euler('xyz',p[3:],degrees=True).as_matrix();return T
def to_wpr(T):
    return np.r_[T[:3,3]*1000,Rotation.from_matrix(T[:3,:3]).as_euler('xyz',degrees=True)].tolist()
def errors(A,B):
    return float(np.linalg.norm(A[:3,3]-B[:3,3])*1000),float(np.rad2deg(Rotation.from_matrix(A[:3,:3].T@B[:3,:3]).magnitude()))
def solve(T,seed):
    def residual(q):
        current=fk(q)
        return np.r_[current[:3,3]-T[:3,3],.2*Rotation.from_matrix(T[:3,:3].T@current[:3,:3]).as_rotvec()]
    r=least_squares(residual,np.clip(seed,LIMITS[:,0]+1e-7,LIMITS[:,1]-1e-7),bounds=LIMITS.T,
        xtol=1e-10,gtol=1e-10,ftol=1e-10,max_nfev=250)
    return r.x,errors(fk(r.x),T)
