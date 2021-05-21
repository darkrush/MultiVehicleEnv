import numpy as np

def coord_dist(coord_a,coord_b):
    return ((coord_a[0]-coord_b[0])**2 + (coord_a[1]-coord_b[1])**2)**0.5

def naive_inference(tx,ty,theta,dist=0.0,min_r=0.0):
    r = (tx**2+ty**2)**0.5
    relate_theta = np.arctan2(ty,tx)-theta
    yt = np.sin(relate_theta)*r
    xt = np.cos(relate_theta)*r
    if abs(np.tan(relate_theta)*r) < dist * 0.5:
        vel = np.sign(xt)
        phi = 0
    else:
        in_min_r = (xt**2+(abs(yt)-min_r)**2)< min_r**2
        vel = -1 if (bool(in_min_r) ^ bool(xt<0)) else 1
        phi = -1 if (bool(in_min_r) ^ bool(yt<0)) else 1
    return vel,phi