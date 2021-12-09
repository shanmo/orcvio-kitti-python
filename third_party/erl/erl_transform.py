import numpy as np
import transforms3d as t3d


def restrictAngle(a, minrange=-np.pi,maxrange=np.pi):
  '''
  a = restrictAngle(a,minrange,maxrange)

  restricts the angles a to the range [minrange,maxrange]
  
  @Input:
    a = n x 1   
  '''
  return minrange + (a - minrange)%(maxrange - minrange)


def angularDistance(a,b):
  '''
  returns the angular distance between two angles a and b in RAD
  
  @Input:
    a,b = n x 1
  '''
  return np.abs(restrictAngle(a-b))
  

def angularMean(a,w):
  ''' m = angularMean(ang,w)

     a = n x 1 = vector of angles in [rad]
     w = n x 1 = vector of weights to compute weighted mean
     m = 1x1 = average angle in [rad]
  '''  
  return np.arctan2( np.dot(np.sin(a),w), np.dot(np.cos(a),w) )
  
  
def rotx(a):
  ''' a rotation of a about the X axis'''
  ca, sa = np.cos(a), np.sin(a)
  zz, ee = np.zeros_like(a), np.ones_like(a)
  R = np.empty(a.shape+(3,3)) if type(a) is np.ndarray else np.empty((3,3))
  R[...,0,0], R[...,0,1], R[...,0,2] = ee, zz, zz
  R[...,1,0], R[...,1,1], R[...,1,2] = zz, ca,-sa
  R[...,2,0], R[...,2,1], R[...,2,2] = zz, sa, ca
  return R

def roty(a):
  ''' a rotation of a about the Y axis'''
  ca, sa = np.cos(a), np.sin(a)
  zz, ee = np.zeros_like(a), np.ones_like(a)
  R = np.empty(a.shape+(3,3)) if type(a) is np.ndarray else np.empty((3,3))
  R[...,0,0], R[...,0,1], R[...,0,2] = ca, zz, sa
  R[...,1,0], R[...,1,1], R[...,1,2] = zz, ee, zz
  R[...,2,0], R[...,2,1], R[...,2,2] =-sa, zz, ca
  return R

def rotz(a):
  ''' a rotation of a about the Z axis'''
  ca, sa = np.cos(a), np.sin(a)
  zz, ee = np.zeros_like(a), np.ones_like(a)
  R = np.empty(a.shape+(3,3)) if type(a) is np.ndarray else np.empty((3,3))
  R[...,0,0], R[...,0,1], R[...,0,2] = ca,-sa, zz
  R[...,1,0], R[...,1,1], R[...,1,2] = sa, ca, zz
  R[...,2,0], R[...,2,1], R[...,2,2] = zz, zz, ee
  return R

def rotxh(a):
  ''' a rotation of a about the X axis in homogeneous coordinates'''
  ca, sa = np.cos(a), np.sin(a)
  zz, ee = np.zeros_like(a), np.ones_like(a)
  R = np.empty(a.shape+(4,4)) if type(a) is np.ndarray else np.empty((4,4))
  R[...,0,0], R[...,0,1], R[...,0,2], R[...,0,3] = ee, zz, zz, zz
  R[...,1,0], R[...,1,1], R[...,1,2], R[...,1,3] = zz, ca,-sa, zz
  R[...,2,0], R[...,2,1], R[...,2,2], R[...,2,3] = zz, sa, ca, zz
  R[...,3,0], R[...,3,1], R[...,3,2], R[...,3,3] = zz, zz, zz, ee
  return R

def rotyh(a):
  ''' a rotation of a about the Y axis in homogeneous coordinates'''
  ca, sa = np.cos(a), np.sin(a)
  zz, ee = np.zeros_like(a), np.ones_like(a)
  R = np.empty(a.shape+(4,4)) if type(a) is np.ndarray else np.empty((4,4))
  R[...,0,0], R[...,0,1], R[...,0,2], R[...,0,3] = ca, zz, sa, zz
  R[...,1,0], R[...,1,1], R[...,1,2], R[...,1,3] = zz, ee, zz, zz
  R[...,2,0], R[...,2,1], R[...,2,2], R[...,2,3] =-sa, zz, ca, zz
  R[...,3,0], R[...,3,1], R[...,3,2], R[...,3,3] = zz, zz, zz, ee
  return R

def rotzh(a):
  ''' a rotation of a about the Z axis in homogeneous coordinates'''
  ca, sa = np.cos(a), np.sin(a)
  zz, ee = np.zeros_like(a), np.ones_like(a)
  R = np.empty(a.shape+(4,4)) if type(a) is np.ndarray else np.empty((4,4))
  R[...,0,0], R[...,0,1], R[...,0,2], R[...,0,3] = ca,-sa, zz, zz
  R[...,1,0], R[...,1,1], R[...,1,2], R[...,1,3] = sa, ca, zz, zz
  R[...,2,0], R[...,2,1], R[...,2,2], R[...,2,3] = zz, zz, ee, zz
  R[...,3,0], R[...,3,1], R[...,3,2], R[...,3,3] = zz, zz, zz, ee
  return R

def rot2d(a):
  ''' a rotation of a about the Z axis in 2D
      a = n x 1
      R = n x 2 x 2
  '''
  ca, sa = np.cos(a), np.sin(a)
  R = np.empty(a.shape+(2,2)) if type(a) is np.ndarray else np.empty((2,2))
  R[...,0,0], R[...,0,1] = ca,-sa
  R[...,1,0], R[...,1,1] = sa, ca
  return R

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

def rot3d(yaw,pitch,roll,axes='rzyx'):
  '''
    a rotation of yaw, pitch, roll in the default sequence: 'rZYX'
  
    yaw = n x 1
    pitch = n x 1
    roll = n x 1
    R = n x 3 x 3
  '''
  firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
  i = firstaxis
  j = [1, 2, 0, 1][i+parity]   # axis sequences for Euler angles
  k = [1, 2, 0, 1][i-parity+1] # axis sequences for Euler angles

  if frame:
    yaw, roll = roll, yaw
  if parity:
    yaw, pitch, roll = -yaw, -pitch, -roll
  
  sy, sp, sr = np.sin(yaw), np.sin(pitch), np.sin(roll)
  cy, cp, cr = np.cos(yaw), np.cos(pitch), np.cos(roll)
  cycr, cysr = cy*cr, cy*sr
  sycr, sysr = sy*cr, sy*sr
  R = np.empty(yaw.shape+(3,3)) if type(yaw) is np.ndarray else np.empty((3,3))
  if repetition:
    R[...,i,i], R[...,i,j], R[...,i,k] = cp, sp*sy, sp*cy
    R[...,j,i], R[...,j,j], R[...,j,k] = sp*sr,-cp*sysr + cycr,-cp*cysr - sycr
    R[...,k,i], R[...,k,j], R[...,k,k] =-sp*cr, cp*sycr + cysr, cp*cycr - sysr
  else:
    R[...,i,i], R[...,i,j], R[...,i,k] = cp*cr, sp*sycr-cysr, sp*cycr+sysr
    R[...,j,i], R[...,j,j], R[...,j,k] = cp*sr, sp*sysr+cycr, sp*cysr-sycr
    R[...,k,i], R[...,k,j], R[...,k,k] = -sp, cp*sy, cp*cy
  return R



def axangle2tangentquat(a):
  '''
  converts an n x 3 axis-angle to an n x 4 tangent space quaternion 
  '''
  q = np.empty(a.shape[:-1]+(4,))
  q[...,0].fill(0)
  q[...,1:] = a/2.0
  return q

def quatExp(q):
  '''
  returns the exponential function of the quaternion q
  assumes q is Mx4 dimensional Hamilton quaternion (q[...,0] = scalar)
  '''
  qv = q[...,1:] # M x 3
  norm_v = np.sqrt(np.sum(qv**2,-1)) # M x 1
  svnv = np.nan_to_num(np.sin(norm_v)/norm_v) # M x 1
  exp_q =  np.empty_like(q)
  exp_q[...,0], exp_q[...,1:] = np.cos(norm_v), qv * svnv[...,None]
  #exp_q = np.hstack((np.cos(norm_v),qv * svnv)) # M x 4
  exp_q = exp_q * np.exp(q[...,0,None])
  return exp_q
  
def quat2rot(q):
  '''
  converts the Nx4 Hamilton quaternion q to a Nx3x3 rotation matrix R
  '''
  R = np.empty(q.shape[:-1]+(3,3))
  R[...,0,0] = 1-2*(q[...,2]**2+q[...,3]**2)
  R[...,0,1] = 2*q[...,1]*q[...,2]-2*q[...,0]*q[...,3]
  R[...,0,2] = 2*q[...,0]*q[...,2]+2*q[...,1]*q[...,3]
  R[...,1,0] = 2*q[...,1]*q[...,2]+2*q[...,0]*q[...,3]
  R[...,1,1] = 1-2*(q[...,1]**2+q[...,3]**2)
  R[...,1,2] = 2*q[...,2]*q[...,3]-2*q[...,0]*q[...,1]
  R[...,2,0] = 2*q[...,1]*q[...,3]-2*q[...,0]*q[...,2]
  R[...,2,1] = 2*q[...,2]*q[...,3]+2*q[...,0]*q[...,1]
  R[...,2,2] = 1-2*(q[...,1]**2+q[...,2]**2)
  return R
  
  #R = np.stack((1-2*(q[...,2]**2+q[...,3]**2),\
  #              2*q[...,1]*q[...,2]-2*q[...,0]*q[...,3],\
  #              2*q[...,0]*q[...,2]+2*q[...,1]*q[...,3],\
  #              2*q[...,1]*q[...,2]+2*q[...,0]*q[...,3],\
  #              1-2*(q[...,1]**2+q[...,3]**2),\
  #              2*q[...,2]*q[...,3]-2*q[...,0]*q[...,1],\
  #              2*q[...,1]*q[...,3]-2*q[...,0]*q[...,2],\
  #              2*q[...,2]*q[...,3]+2*q[...,0]*q[...,1],\
  #              1-2*(q[...,1]**2+q[...,2]**2)),axis=-1) # N x 9
  #return R.reshape(R.shape[:-1] + (3,3))
  
def quat2rotPost(q):
  '''
  converts the 4xN Hamilton quaternion q to a 3x3xN rotation matrix R
  '''
  R = np.zeros((3,3,q.shape[0]))
  R[:,0,0] = 1-2*(q[2,:]**2+q[3,:]**2)
  R[:,0,1,:] = 2*q[1,:]*q[2,:]-2*q[0,:]*q[3,:]
  R[:,0,2,:] = 2*q[0,:]*q[2,:]+2*q[1,:]*q[3,:]
  
  R[:,1,0,:] = 2*q[1,:]*q[2,:]+2*q[0,:]*q[3,:]
  R[:,1,1,:] = 1-2*(q[1,:]**2+q[3,:]**2)
  R[:,1,2,:] = 2*q[2,:]*q[3,:]-2*q[0,:]*q[1,:]
  
  R[:,2,0,:] = 2*q[1,:]*q[3,:]-2*q[0,:]*q[2,:]
  R[:,2,1,:] = 2*q[2,:]*q[3,:]+2*q[0,:]*q[1,:]
  R[:,2,2,:] = 1-2*(q[1,:]**2+q[2,:]**2)
  return R

def qjpl2rot(q):
  '''
  converts the Mx4 JPL quaternion q to a Mx3x3 rotation matrix R
  '''
  R = np.stack((q[...,0]**2 - q[...,1]**2 - q[...,2]**2 + q[...,3]**2,\
                2*( q[...,0]*q[...,1] + q[...,2]*q[...,3] ),\
                2*( q[...,0]*q[...,2] - q[...,1]*q[...,3] ),\
                2*( q[...,0]*q[...,1] - q[...,2]*q[...,3] ),\
                -q[...,0]**2 + q[...,1]**2 - q[...,2]**2 + q[...,3]**2,\
                2*( q[...,1]*q[...,2] + q[...,0]*q[...,3] ),\
                2*( q[...,0]*q[...,2] + q[...,1]*q[...,3] ),\
                2*( q[...,1]*q[...,2] - q[...,0]*q[...,3] ),\
                -q[...,0]**2 - q[...,1]**2 + q[...,2]**2 + q[...,3]**2),axis=-1) # N x 9
  return R.reshape(R.shape[:-1] + (3,3))
  
def qjpl2rotPost(q):
  '''
  converts the 4xM JPL quaternion q to a 3x3xM rotation matrix R
  '''
  R = np.zeros((3,3,q.shape[1]))
  R[0,0,:] = q[0,:]**2 - q[1,:]**2 - q[2,:]**2 + q[3,:]**2
  R[0,1,:] = 2*( q[0,:]*q[1,:] + q[2,:]*q[3,:] ) 
  R[0,2,:] = 2*( q[0,:]*q[2,:] - q[1,:]*q[3,:] )
 
  R[1,0,:] = 2*( q[0,:]*q[1,:] - q[2,:]*q[3,:] )
  R[1,1,:] = -q[0,:]**2 + q[1,:]**2 - q[2,:]**2 + q[3,:]**2
  R[1,2,:] = 2*( q[1,:]*q[2,:] + q[0,:]*q[3,:] )
 
  R[2,0,:] = 2*( q[0,:]*q[2,:] + q[1,:]*q[3,:] )
  R[2,1,:] = 2*( q[1,:]*q[2,:] - q[0,:]*q[3,:] )
  R[2,2,:] = -q[0,:]**2 - q[1,:]**2 + q[2,:]**2 + q[3,:]**2
  return R





def axangle2skew(a):
  '''
  converts an n x 3 axis-angle to an n x 3 x 3 skew symmetric matrix 
  '''
  S = np.empty(a.shape[:-1]+(3,3))
  S[...,0,0].fill(0)
  S[...,0,1] =-a[...,2]
  S[...,0,2] = a[...,1]
  S[...,1,0] = a[...,2]
  S[...,1,1].fill(0)
  S[...,1,2] =-a[...,0]
  S[...,2,0] =-a[...,1]
  S[...,2,1] = a[...,0]
  S[...,2,2].fill(0)
  return S

def skew2axangle(S):
  '''
  converts an n x 3 x 3 skew symmetric matrix to an n x 3 axis-angle 
  '''
  return S[...,[2,0,1],[1,2,0]]
  
def skew2rot(S):
  '''
  converts an n x 3 x 3 skew symmetric (so3) matrix to an n x 3 x 3 rotation (SO3) matrix 
  '''  
  return axangle2rot(skew2axangle(S))

def axangle2rot(a):
  '''
  @Input:
    a = n x 3 = n axis-angle elements 
  @Output:
    R = n x 3 x 3 = n elements of SO(3)
  '''
  na = np.linalg.norm(a,axis=-1) # n x 1
  ana = np.nan_to_num(a/na[...,None]) # n x 3
  ca, sa = np.cos(na), np.sin(na) # n x 1
  mc_ana = ana * (1 - ca[...,None]) # n x 3
  sa_ana = ana * sa[...,None] # n x 3
  
  R = np.empty(a.shape+(3,))
  R[...,0,0] = mc_ana[...,0]*ana[...,0] + ca
  R[...,0,1] = mc_ana[...,0]*ana[...,1] - sa_ana[...,2]
  R[...,0,2] = mc_ana[...,0]*ana[...,2] + sa_ana[...,1]
  R[...,1,0] = mc_ana[...,0]*ana[...,1] + sa_ana[...,2]
  R[...,1,1] = mc_ana[...,1]*ana[...,1] + ca
  R[...,1,2] = mc_ana[...,2]*ana[...,1] - sa_ana[...,0]
  R[...,2,0] = mc_ana[...,0]*ana[...,2] - sa_ana[...,1]
  R[...,2,1] = mc_ana[...,1]*ana[...,2] + sa_ana[...,0]
  R[...,2,2] = mc_ana[...,2]*ana[...,2] + ca
  return R


	 
def SO3RightJacobian( theta ):
  '''
  theta = n x 3 = axangle
  J = n x 3 x 3
  '''
  theta_norm2 = np.sum(theta**2,axis=-1)
  theta_norm = np.sqrt(theta_norm2)
  theta_norm3 = theta_norm*theta_norm2
  theta_hat = axangle2skew(theta)
  eye = np.zeros_like(theta_hat)
  eye[...,[0,1,2],[0,1,2]] = 1.0
  return eye - np.nan_to_num((1.0 - np.cos(theta_norm))/theta_norm2)*theta_hat \
             + np.nan_to_num((theta_norm - np.sin(theta_norm))/theta_norm3)*theta_hat @ theta_hat

def SO3RightJacobianInverse( theta ):
  '''
  theta = n x 3 = axangle
  invJ = n x 3 x 3 
  '''
  theta_norm2 = np.sum(theta**2,axis=-1)
  theta_norm = np.sqrt(theta_norm2)
  theta_hat = axangle2skew(theta)
  eye = np.zeros_like(theta_hat)
  eye[...,[0,1,2],[0,1,2]] = 1.0  
  return eye + theta_hat/2.0 + np.nan_to_num(1.0/theta_norm2 - (1.0 + np.cos(theta_norm))/2.0/theta_norm/np.sin(theta_norm))*theta_hat @ theta_hat # n x 3 x 3


def SO3LeftJacobian( theta ): 
  '''
  theta = n x 3 = axangle
  J = n x 3 x 3
  '''
  theta_norm2 = np.sum(theta**2,axis=-1)
  theta_norm = np.sqrt(theta_norm2)
  theta_norm3 = theta_norm*theta_norm2
  theta_hat = axangle2skew(theta)
  eye = np.zeros_like(theta_hat)
  eye[...,[0,1,2],[0,1,2]] = 1.0
  return eye + np.nan_to_num((1.0 - np.cos(theta_norm))/theta_norm2)*theta_hat \
             + np.nan_to_num((theta_norm - np.sin(theta_norm))/theta_norm3)*theta_hat @ theta_hat

def SO3LeftJacobianInverse( theta ):
  '''
  theta = n x 3 = axangle
  invJ = n x 3 x 3 
  '''
  theta_norm2 = np.sum(theta**2,axis=-1)
  theta_norm = np.sqrt(theta_norm2)
  theta_hat = axangle2skew(theta)
  eye = np.zeros_like(theta_hat)
  eye[...,[0,1,2],[0,1,2]] = 1.0  
  return np.eye - theta_hat/2.0 + np.nan_to_num(1.0/theta_norm2 - (1.0 + np.cos(theta_norm))/2.0/theta_norm/np.sin(theta_norm))*theta_hat @ theta_hat






def point2homo(p):
  '''
  returns the nx4 or nx3 homogeneous coordinates of the nx3 or nx2 points p
  '''
  ee = np.ones(p.shape[:-1]+(1,))
  return np.concatenate((p,ee),axis=-1)

def homo2point(ph):
  ''' returns the nx3 (or nx2) Cartesian coordinates of the nx4 (or nx3) homogeneous points ph '''
  return ph[...,:-1]/ph[...,-1,None]

def normalizePointh(ph):
  '''
  force the last element of an nx3 or nx4 points in homogeneous coordiantes to be 1
  '''
  return ph/ph[...,-1,None]

def odot(ph):
  '''
  @Input:
	  ph = n x 4 = points in homogeneous coordinates
	@Output:
    odot(ph) = n x 4 x 6
	'''
  zz = np.zeros(ph.shape + (6,))
  zz[...,:3,3:6] = -axangle2skew(ph[...,:3])
  zz[...,0,0],zz[...,1,1],zz[...,2,2] = ph[...,3],ph[...,3],ph[...,3]
  return zz

def circledCirc(ph):
  '''
  @Input:
	  ph = n x 4 = points in homogeneous coordinates
	@Output:
    circledCirc(ph) = n x 6 x 4
	'''
  zz = np.zeros(ph.shape[:-1] + (6,4))
  zz[...,3:,:3] = -axangle2skew(ph[...,:3])
  zz[...,:3,3] = ph[...,:3]
  return zz


def pointh2lineh(a,b):
  '''
  convert two 2D points (lines) in homogenous coordinates to a line (point)
  that passes through them, l = a x b
  a = nx3
  b = nx3
  l = nx3
  '''
  return np.cross(a,b)


def bbox2lineh(bbox):
  '''
  : bbox = n x [left (x_min), up (y_min), right (x_max), down (y_max)]
  : line = n x 4 x 3
  '''
  line = np.empty(bbox.shape[:-1]+(4,3))  
  line[...,0,:] = pointh2lineh(point2homo(bbox[...,[0,1]]),point2homo(bbox[...,[2,1]]))
  line[...,1,:] = pointh2lineh(point2homo(bbox[...,[2,1]]),point2homo(bbox[...,[2,3]]))
  line[...,2,:] = pointh2lineh(point2homo(bbox[...,[2,3]]),point2homo(bbox[...,[0,3]]))
  line[...,3,:] = pointh2lineh(point2homo(bbox[...,[0,3]]),point2homo(bbox[...,[0,1]]))
  return line


def projection(ph):
  '''
  ph = n x 4 = homogeneous point coordinates
  r = n x 4 = ph/ph[...,2] = normalized z axis coordinates
  '''  
  return ph/ph[...,2,None]
  
def projectionJacobian(ph):
  '''
  ph = n x 4 = homogeneous point coordinates
  J = n x 4 x 4 = Jacobian of ph/ph[...,2]
  '''  
  J = np.zeros(ph.shape+(4,))
  iph2 = 1.0/ph[...,2]
  ph2ph2 = ph[...,2]**2
  J[...,0,0], J[...,1,1],J[...,3,3] = iph2,iph2,iph2
  J[...,0,2] = -ph[...,0]/ph2ph2
  J[...,1,2] = -ph[...,1]/ph2ph2
  J[...,3,2] = -ph[...,3]/ph2ph2
  return J

def points2bbox(pts):
  '''
    pts = n x d
    bbox = [x_min, y_min, z_min, x_max, y_max, z_max]
  '''
  bbox = np.concatenate(( np.min(pts,axis=0), np.max(pts,axis=0) ))
  return bbox

def bboxCenter(bbox):
  '''
    bbox = n x [left (x_min), up (y_min), right (x_max), down (y_max)]
  '''
  return (bbox[...,:2] + bbox[..., 2:])/2.0

def conic2bbox(Q):
  '''
  : Q = n x 3 x 3
  : bbox = n x [left (x_min), up (y_min), right (x_max), down (y_max)]
  '''  

  bbox = np.empty(Q.shape[:-2]+(4,))
  A, c = homo2ellipsoid(Q)
  detA = A[...,0,0]*A[...,1,1] - A[...,0,1]**2
  halfw = np.sqrt(A[...,1,1]/detA)
  halfh = np.sqrt(A[...,0,0]/detA)
  #xc = Q[...,0,2] / Q[...,2,2]
  #yc = Q[...,1,2] / Q[...,2,2]
  #halfw = np.sqrt(xc ** 2 - Q[...,0,0])
  #halfh = np.sqrt(yc ** 2 - Q[...,1,1])
  bbox[...,0] = c[...,0] - halfw
  bbox[...,1] = c[...,1] - halfh
  bbox[...,2] = c[...,0] + halfw
  bbox[...,3] = c[...,1] + halfh
  return bbox

def homo2ellipsoid(Q):
  '''
  extract (x-c)'A(x-c) <= 1 from  homo(x)' Q homo(x) <= 0
  '''
  d = Q.shape[-1]-1
  A = Q[...,:d,:d]
  c = -np.linalg.solve(A,Q[...,:d,d])
  return A/(-(Q[...,d,:d]*c).sum(-1)-Q[...,d,d])[...,None,None], c
  
def ellipsoid2homo(A,c):
  '''
  extract homo(x)' Q homo(x) <= 0 from (x-c)'A(x-c) <= 1
  '''
  d = A.shape[-1]
  Q = np.empty(A.shape[:-2]+(d+1,d+1))
  Q[...,:d,:d] = A
  Q[...,:d, d] = -A@c
  Q[..., d,:d] = Q[...,:d,-1]
  Q[..., d, d] = -(c*Q[...,:d, d]).sum(-1) - 1.0
  return Q
  
def normalizeQuadric(Q):
  '''
  Q = n x 3 x 3 or n x 4 x 4
  '''
  return Q/Q[...,-1,-1,None,None]  
  

def transformDualQuadrics(Q, T, flag = False):
  '''
  * @brief: Converts the dual quadric(s) Q from one frame to another using the transformation or projection T.
  *         Mote that if T is a 3 x 4 monocular or 6 x 4 stereo projection, then Q is transformed to a conic.
  *
  * @Input:
  *    T = n x p x 4 = 4x4 transformation or 3x4 projection or 6x4 projection 
  *    Q = m x 4 x 4 = quadrics
  *
  * Transformation (4x4):
  *    T = [ R, p ]
  *        [ 0, 1 ]
  *
  * Mono Projection (3x4):
  *    T = [ f*su, f*st, cu, 0]
  *        [    0, f*sv, cv, 0]
  *        [    0,    0,  1, 0]
  *
  * Stereo Projection (6x4):
  *    T = [ f*su, f*st, cu,       0]
  *        [    0, f*sv, cv,       0]
  *        [    0,    0,  1,       0]
  *        [ f*su, f*st, cu, -f*su*b]
  *        [    0, f*sv, cv,       0]
  *        [    0,    0,  1,       0]
  *
  *   where
  *     f = focal scaling in meters
  *     su = pixels per meter
  *     sv = pixels per meter
  *     cu = image center
  *     cv = image center
  *     st = nonrectangular pixel scaling
  *      b = baseline
  *
  *   flag = if true returns the transformation of every Q versus every T
  *          else assumes n = m and returns the transformations for the matching elements
  *
  * @Output:
  *   C = transformed Q = max(m, n) x p x p  if flag = False
  *                       n x m x p x p      if flag = True  
  '''
  if flag: # 1 < m != n > 1
    # (n x m x p x 4) @ (n x 1 x 4 x p)
    C = T[...,None,:,:]@Q[None,...]@np.swapaxes(T,-1,-2)[...,None,:,:]
  else:
    C = T@Q@np.swapaxes(T,-1,-2)
  return np.squeeze(C)


def transformPoints(P, T, flag = False):
  '''
  * @brief: Returns the transformed coordinates of 3D point(s) P in homogeneous coordinates
  *         using the transformation or projection T. Mote that if T is a 3 x 4 monocular or
  *         6 x 4 stereo projection, then P is transformed to 2D.
  *  
  * @Input:
  *    T = n x p x 4 = 4x4 transformation or 3x4 mono projection or 6x4 stereo projection 
  *    P = m x 4 = points
  *
  *
  * Transformation (4x4):
  *    T = [ R, p ]
  *        [ 0, 1 ]
  *
  * Mono Projection (3x4):
  *    T = [ f*su, f*st, cu, 0]
  *        [    0, f*sv, cv, 0]
  *        [    0,    0,  1, 0]
  *
  * Stereo Projection (6x4):
  *    T = [ f*su, f*st, cu,       0]
  *        [    0, f*sv, cv,       0]
  *        [    0,    0,  1,       0]
  *        [ f*su, f*st, cu, -f*su*b]
  *        [    0, f*sv, cv,       0]
  *        [    0,    0,  1,       0]
  *
  *   where
  *     f = focal scaling in meters
  *     su = pixels per meter
  *     sv = pixels per meter
  *     cu = image center
  *     cv = image center
  *     st = nonrectangular pixel scaling
  *      b = baseline
  *
  *   flag = if true returns the transformation of every P versus every T
  *          else assumes n = m and returns the transformations for the matching elements
  *
  * @Output:
  *    M = transformed P = max(m, n) x p if flag = False
  *                        n x m x p if flag = True
  *
  ''' 
  if flag: # 1 < m != n > 1
    M = T[...,None,:,:]@P[None,...,None] # n x m x p x 1
  else:
    M = T@P[...,None] # max(m, n) x p x 1
  return np.squeeze(M)



## Poses

def inversePose(T):
  '''
  @Input:
    T = n x 4 x 4 = n elements of SE(3)
  @Output:
    iT = n x 4 x 4 = inverse of T
  '''
  iT = np.empty_like(T)
  iT[...,0,0], iT[...,0,1], iT[...,0,2] = T[...,0,0], T[...,1,0], T[...,2,0] 
  iT[...,1,0], iT[...,1,1], iT[...,1,2] = T[...,0,1], T[...,1,1], T[...,2,1] 
  iT[...,2,0], iT[...,2,1], iT[...,2,2] = T[...,0,2], T[...,1,2], T[...,2,2]
  iT[...,:3,3] = -np.squeeze(iT[...,:3,:3] @ T[...,:3,3,None])
  iT[...,3,:] = T[...,3,:]
  return iT


def axangle2twist(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    T = n x 4 x 4 = n elements of se(3)
  '''
  T = np.zeros(x.shape[:-1]+(4,4))
  T[...,0,1] =-x[...,5]
  T[...,0,2] = x[...,4]
  T[...,0,3] = x[...,0]
  T[...,1,0] = x[...,5]
  T[...,1,2] =-x[...,3]
  T[...,1,3] = x[...,1]
  T[...,2,0] =-x[...,4]
  T[...,2,1] = x[...,3]
  T[...,2,3] = x[...,2]
  return T

def twist2axangle(T):
  '''
  converts an n x 4 x 4 twist (se3) matrix to an n x 6 axis-angle 
  '''
  return T[...,[0,1,2,2,0,1],[3,3,3,1,2,0]]

def axangle2adtwist(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    A = n x 6 x 6 = n elements of ad(se(3))
  '''
  A = np.zeros(x.shape+(6,))
  A[...,0,1] =-x[...,5]
  A[...,0,2] = x[...,4]
  A[...,0,4] =-x[...,2]
  A[...,0,5] = x[...,1]
  
  A[...,1,0] = x[...,5]
  A[...,1,2] =-x[...,3]
  A[...,1,3] = x[...,2]
  A[...,1,5] =-x[...,0]
  
  A[...,2,0] =-x[...,4]
  A[...,2,1] = x[...,3]
  A[...,2,3] =-x[...,1]
  A[...,2,4] = x[...,0]
  
  A[...,3,4] =-x[...,5] 
  A[...,3,5] = x[...,4] 
  A[...,4,3] = x[...,5]
  A[...,4,5] =-x[...,3]   
  A[...,5,3] =-x[...,4]
  A[...,5,4] = x[...,3]
  return A

def twist2pose(T):
  '''
  converts an n x 4 x 4 twist (se3) matrix to an n x 4 x 4 pose (SE3) matrix 
  '''
  rotang2 = np.sum(T[...,[2,0,1],[1,2,0]]**2,axis=-1)[...,None,None] # n x 1
  rotang = np.sqrt(rotang2)
  rotang3 = rotang * rotang2
  T2 = T@T
  T3 = T@T2
  eye = np.zeros_like(T)
  eye[...,[0,1,2,3],[0,1,2,3]] = 1.0
  return eye + T + np.nan_to_num((1.0 - np.cos(rotang))/rotang2*T2 + (rotang - np.sin(rotang))/rotang3*T3)

def axangle2pose(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    T = n x 4 x 4 = n elements of SE(3)
  '''
  return twist2pose(axangle2twist(x))

def pose2adpose(T):
  '''
  converts an n x 4 x 4 pose (SE3) matrix to an n x 6 x 6 adjoint pose (ad(SE3)) matrix 
  '''
  calT = np.empty(T.shape[:-2]+(6,6))
  calT[...,:3,:3] = T[...,:3,:3]
  calT[...,:3,3:] = axangle2skew(T[...,:3,3]) @ T[...,:3,:3]
  calT[...,3:,:3] = np.zeros(T.shape[:-2]+(3,3))
  calT[...,3:,3:] = T[...,:3,:3]
  return calT


def findRotation( S1, S2 ):
  '''
   * Returns the rotation matrix R that minimizes the error S1 - R*S2
   * S1, S2 are nx3 matrices of corresponding points
  '''
  M = S1.T @ S2 # 3 x 3
  U, _, VH = np.linalg.svd(M, full_matrices=False)
  return U @ np.diag([1.0,1.0,np.linalg.det(U @ VH)]) @ VH

def findTransform( S1, S2 ):
  '''
   * Find the rotation and translation aligning the points  S2 (nx3) to S1 (nx3)
  '''
  S1m = S1.mean(axis=0,keepdims=True)
  S1c = S1 - S1m
  S2m = S2.mean(axis=0,keepdims=True)
  S2c = S2 - S2m
  R = findRotation( S1c, S2c )
  return np.squeeze(S1m - S2m @ R.T), R

def triangulateFrom2Views(T,z1,z2):
  '''
  % @INPUT:
  %   T = transformation from frame 1 to frame 2 between 
  %       two camera frames observing the same features
  %   z1,z2 = m x 2 = feature observations in the two camera frames
  %
  % @OUTPUT
  %   y = m x 3 = feature position computed in frame 1
  %
  '''
  # Construct a least squares problem
  m = z1@T[:3,:2].T # m x 3
  ax = m[...,0]+T[0,2]-z2[...,0]*(m[...,2]+T[2,2])
  ay = m[...,1]+T[1,2]-z2[...,1]*(m[...,2]+T[2,2])
  bx = z2[...,0]*T[2,3] - T[0,3]
  by = z2[...,1]*T[2,3] - T[1,3]
  depth = (ax*bx+ay*by)/(ax*ax+ay*ay)
  return np.array([z1[...,0]*depth,z1[...,1]*depth,depth]).T


def findSimilarityTransform( S1, S2 ):
  '''
   * Finds a full similarity transform including scaling aligning the points S2 (nx3) to S1 (nx3)
   * The returned transformation is from S2 to S1
  '''
  S1m = S1.mean(axis=0,keepdims=True)
  S1c = S1 - S1m
  S2m = S2.mean(axis=0,keepdims=True)
  S2c = S2 - S2m
  R = findRotation( S1c, S2c )  
  S2c = S2c @ R.T # n x 3
  w = (S1c.T @ S2c).trace() / (S2c.T @ S2c).trace()
  return np.squeeze(S1m - w * S2m @ R.T), R, w
  
def projPoint2Line(m,a,b):
  '''
  Returns:
  - coordinates of the projection of M (nx3) on AB (each nx3)
  - the distance d0 between M and P
  - the distance d1 between A and P
  - the distance d2 between B and P  
  '''
  ab = b-a
  am = m-a
  ab_norm = np.sqrt(np.sum(ab**2,axis=-1))
  ab_dot_am = np.sum(ab*am,axis=-1)
  d1 = ab_dot_am / ab_norm
  d2 = ab_norm - d1
  p = ab/d1[...,None]-a
  mp = p-m
  d0 = np.sqrt(np.sum(mp**2,axis=-1))
  return p, d0, d1, d2

 
















# def skewSymmetricMatrix(x):
  # '''
    # x = n x 3 = n elements of axis-angle
    # hat(x) = n x 3 x 3 = n skew symmetric matrices corresponding to x
  # '''
  # x0 = x[...,0]
  # x1 = x[...,1]
  # x2 = x[...,2]
  # zz = np.zeros_like(x0)
  # hx = np.stack((zz,-x2,x1,x2,zz,-x0,-x1,x0,zz),axis=-1) # n x 9
  # return R.reshape(R.shape[:-1] + (3,3))
  # #return np.stack(( np.stack((zz,-x2,x1)), np.stack((x2,zz,-x0)), np.stack((-x1,x0,zz)) ))


# def skewSymmetricMatrixPost(x):
  # x0 = x[0]
  # x1 = x[1]
  # x2 = x[2]
  # zz = np.zeros_like(x0)
  # return np.stack(( np.stack((zz,-x2,x1)), np.stack((x2,zz,-x0)), np.stack((-x1,x0,zz)) ))

# def skewSymmetricMatrixJacobian():
  # return np.array([[0.0,  0.0,  0.0],\
	        # [0.0,  0.0,  1.0],\
	        # [0.0, -1.0,  0.0],\
	        # [0.0,  0.0, -1.0],\
	        # [0.0,  0.0,  0.0],\
	        # [1.0,  0.0,  0.0],\
	        # [0.0,  1.0,  0.0],\
	       # [-1.0,  0.0,  0.0],\
	        # [0.0,  0.0,  0.0]])

def se3HatPost(x):
  '''
  @Input:
    x = 6xn = n elements of position and axis-angle
  @Output:
    hx = 4x4xn = n elements of se(3)
  '''
  zz = np.zeros_like(x[0])
  return np.stack(( np.stack((zz,-x[5],x[4],x[0])),\
                    np.stack((x[5],zz,-x[3],x[1])),\
                    np.stack((-x[4],x[3],zz,x[2])),\
                    np.stack((zz,zz,zz,zz)) ))

def se3HatHatPost(x):
  '''
  @Input:
    x = 6xn = n elements of position and axis-angle
  @Output:
    hx = 6x6xn = n elements of ad(se(3))
  '''
  zz = np.zeros_like(x[0])
  return np.stack(( np.stack((zz,-x[5],x[4],zz,-x[2],x[1])),\
                    np.stack((x[5],zz,-x[3],x[2],zz,-x[0])),\
                    np.stack((-x[4],x[3],zz,-x[1],x[0],zz)),\
                    np.stack((zz,zz,zz,zz,-x[5],x[4])),\
                    np.stack((zz,zz,zz,x[5],zz,-x[3])),\
                    np.stack((zz,zz,zz,-x[4],x[3],zz)) )) 

def axisAngleToRotPost( theta ):
  '''
  @Input:
    theta = 3xn = n axis-angle elements 
  @Output:
    R = 3x3xn = n elements of SO(3)
  '''
  nt = np.linalg.norm(theta,axis=0) # 1 x n
  tnt = np.nan_to_num(theta/nt) # 3 x n
  ct = np.cos(nt) # 1 x n
  st = np.sin(nt) # 1 x n
  mc_tnt = tnt * (1 - ct) # 3 x n
  st_tnt = tnt * st # 3 x n
  return np.stack(( np.stack(( mc_tnt[0]*tnt[0]+ct,         mc_tnt[0]*tnt[1]+st_tnt[2],  mc_tnt[0]*tnt[2]-st_tnt[1])),\
                    np.stack(( mc_tnt[0]*tnt[1]-st_tnt[2],  mc_tnt[1]*tnt[1]+ct,         mc_tnt[1]*tnt[2]+st_tnt[0])),\
                    np.stack(( mc_tnt[0]*tnt[2]+st_tnt[1],  mc_tnt[2]*tnt[1]-st_tnt[0],  mc_tnt[2]*tnt[2]+ct)) ))


	 
# def SO3RightJacobian( theta ):
  # theta_norm2 = theta.dot(theta)
  # theta_norm = np.sqrt(theta_norm2)
  # theta_norm3 = theta_norm*theta_norm2
  # theta_hat = skewSymmetricMatrix(theta)
  # return np.eye(3) - (1.0 - np.cos(theta_norm))/theta_norm2*theta_hat \
                   # + (theta_norm - np.sin(theta_norm))/theta_norm3*theta_hat @ theta_hat

# def SO3RightJacobianInverse( theta ):
  # theta_norm2 = theta.dot(theta)
  # theta_norm = np.sqrt(theta_norm2)
  # theta_hat = skewSymmetricMatrix(theta)
  # return np.eye(3) + theta_hat/2.0 + (1.0/theta_norm2 - (1.0 + np.cos(theta_norm))/2.0/theta_norm/np.sin(theta_norm))*theta_hat @ theta_hat # 3 x 3

# def SO3LeftJacobian( theta ): 
  # theta_norm2 = theta.dot(theta)
  # theta_norm = np.sqrt(theta_norm2)
  # theta_norm3 = theta_norm*theta_norm2
  # theta_hat = skewSymmetricMatrix(theta)
  # return np.eye(3) + (1.0 - np.cos(theta_norm))/theta_norm2*theta_hat \
                   # + (theta_norm - np.sin(theta_norm))/theta_norm3*theta_hat @ theta_hat  
  
# def SO3LeftJacobianInverse( theta ):
  # theta_norm2 = theta.dot(theta)
  # theta_norm = np.sqrt(theta_norm2)
  # theta_hat = skewSymmetricMatrix(theta)
  # return np.eye(3) - theta_hat/2.0 + (1.0/theta_norm2 - (1.0 + np.cos(theta_norm))/2.0/theta_norm/np.sin(theta_norm))*theta_hat @ theta_hat


def expSE3(x):
  '''
  @Input:
    x = 6xn = n elements of position and axis-angle
  @Output:
    T = 4x4xn = n elements of SE(3)
  '''
  # rotang = np.linalg.norm(x[3:],axis=0) # 1 x n
  rotang2 = np.einsum('i...,i...',x[3:],x[3:]) # 1 x n
  rotang = np.sqrt(rotang2)
  rotang3 = rotang * rotang2
  xh = se3Hat(x) # 4x4xn
  xh2 = np.einsum('ij...,j...->i...',xh,xh) # 4x4xn
  xh3 = np.einsum('ij...,j...->i...',xh,xh2) # 4x4xn
  eye = np.zeros_like(xh)
  eye[[0,1,2,3],[0,1,2,3]] = 1.0
  return eye + xh +  np.nan_to_num((1.0 - np.cos(rotang))/rotang2*xh2 + (rotang - np.sin(rotang))/rotang3*xh3)

def inverseTransform(T):
  '''
  @Input:
    T = 4x4xn = n elements of SE(3)
  @Output:
    iT = 4x4xn = inverse of T
  '''
  # iR = T[:3,:3,:].transpose([1, 0, 2])
  iR = np.moveaxis(T[:3,:3,...],0,1)
  ip = np.einsum('ij...,j...->i...',-iR,T[:3,3,...])[:,None,...]
  return np.vstack(( np.hstack(( iR, ip)), T[[3],:,...]))
  



  


 

   
