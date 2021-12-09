import numpy as np
from scipy.linalg import block_diag
from numpy.lib.stride_tricks import as_strided
import time 
import yaml
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors


def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took %s sec.\n' % (nm,(time.time() - tstart)))



def kf_predict(mu,f,S,F,W):
  return f(mu), F@S@F.T + W

def kf_update(mu,h,z,S,H,V):
  innovation = z - h(mu)
  HS = H@S
  kalmanGainTranspose = np.linalg.solve(HS@H.T+V,HS)
  return innovation.T.reshape(-1)@kalmanGainTranspose, S - HS.T@kalmanGainTranspose

def eyekron(n,A):
  '''
    Quick computation for K = kron(eye(n),A)
  '''
  K = np.zeros((A.shape[0]*n, A.shape[1]*n), dtype=A.dtype)
  # K.reshape(2,3,2,2).transpose((0,2,1,3))
  kstr = K.strides
  c = as_strided(K, shape=(n,n,A.shape[0],A.shape[1]), strides=(A.shape[0]*kstr[0], A.shape[1]*kstr[1], kstr[0], kstr[1]))
  c[range(n), range(n)] = A
  return K

def eyekron2(n,A):
  '''
    Quick computation for K = kron(eye(n),A)
  '''
  return block_diag(*([A] * n))


def kroneye(A,n):
  '''
    Quick computation for K = np.kron(A,np.eye(n))
  '''
  K = np.zeros((A.shape[0]*n, A.shape[1]*n), dtype=A.dtype)
  kstr = K.strides
  # K.reshape(A.shape[0],n,A.shape[1],n).transpose((1,3,0,2))
  c = as_strided(K, shape=(n,n,A.shape[0],A.shape[1]), strides=(kstr[0],kstr[1],n*kstr[0],n*kstr[1]))
  c[range(n), range(n)] = A
  return K  

  
def read_yaml(yaml_file):
  stream = open(yaml_file, "r")
  docs = yaml.load_all(stream)
  settings = {}
  for doc in docs:
    for k,v in doc.items():
      settings[k] = v
  return settings



def plot_traj(ax, traj):
  ''' h = plot_traj(h,traj,style)
      
      traj = num_pts x num_dim x num_traj
      style = 'b-'
  '''
  if type(ax) is LineCollection:
    ax.set_verts(np.transpose(traj,(2,0,1)))
  else:
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    h = ax.add_collection(LineCollection(np.transpose(traj,(2,0,1)),colors=colors))
    return h

#def plot_traj(ax, traj):
#  ''' h = plot_traj(h,traj,style)
#      
#      traj = num_pts x num_dim x num_traj
#      style = 'b-'
#  '''
#  # ,**kwargs
#  #num_pts = traj.shape[0]
#  #num_traj = traj.shape[2]
#  #if num_pts == 1:
#  #  traj = np.concatenate((traj,traj),axis=0)
#  if type(ax) is Line2D:
#    ax.set_xdata(traj[:,0,:])
#    ax.set_ydata(traj[:,1,:])
#  else:
#    h = ax.plot(traj[:,0,:],traj[:,1,:])
#    return h

def plot_pose(ax, pose, clr = 'red', sz = 0.5):
  ''' poses = [x,y,yaw] '''
  xy = np.array([[pose[0]+sz*np.cos(pose[2]), pose[1]+sz*np.sin(pose[2])],\
                        [pose[0]+sz*np.cos(pose[2]+2.7), pose[1]+sz*np.sin(pose[2]+2.7)],\
                        [pose[0]+sz*np.cos(pose[2]-2.7), pose[1]+sz*np.sin(pose[2]-2.7)]])
  if type(ax) is Polygon:
    ax.set_xy(xy)
  else:
    h = ax.add_patch(Polygon(xy,color=clr))
    return h

  #  X = np.array([[1,1], [2,2.5], [3, 1], [8, 7.5], [7, 9], [9, 9]])
  #  Y = ['red', 'red', 'red', 'blue', 'blue', 'blue']

  #  plt.figure()
  #  plt.scatter(X[:, 0], X[:, 1], s = 170, color = Y[:])

  #  t1 = plt.Polygon(X[:3,:], color=Y[0])
  #  plt.gca().add_patch(t1)

  #  t2 = plt.Polygon(X[3:6,:], color=Y[3])
  #  plt.gca().add_patch(t2)

  #  plt.show()


def plot_frame(ax, p, R, sz = 1):
  '''
  Plots frames in 3D corresponding to 3xN positions p with 3x3xN orientations R
  '''
  
  x1 = np.stack((p[0,:], p[0,:] + sz*R[0,0,:]))
  x2 = np.stack((p[1,:], p[1,:] + sz*R[1,0,:]))
  x3 = np.stack((p[2,:], p[2,:] + sz*R[2,0,:]))
  
  y1 = np.stack((p[0,:], p[0,:] + sz*R[0,1,:]))
  y2 = np.stack((p[1,:], p[1,:] + sz*R[1,1,:]))
  y3 = np.stack((p[2,:], p[2,:] + sz*R[2,1,:]))

  z1 = np.stack((p[0,:], p[0,:] + sz*R[0,2,:]))
  z2 = np.stack((p[1,:], p[1,:] + sz*R[1,2,:]))
  z3 = np.stack((p[2,:], p[2,:] + sz*R[2,2,:]))
  
  import pdb
  pdb.set_trace()
  
  if isinstance(ax,list):
    h[0].set_xdata(x1)
    h[0].set_ydata(x2)
    h[0].set_3d_properties(x3)
    h[1].set_xdata(y1)
    h[1].set_ydata(y2)
    h[1].set_3d_properties(y3)
    h[2].set_xdata(z1)
    h[2].set_ydata(z2)
    h[2].set_3d_properties(z3)     
  else:
    h = ax.plot(x1, x2, x3, 'r--')
    h[1] = ax.plot(y1, y2, y3, 'g--')
    h[2] = ax.plot(z1, z2, z3, 'b--')
    return h
  
  

def odd_ceil_array(x):
  ''' returns the first odd integer larger than x '''
  rndidx =np.abs( np.floor(x) - x ) <= np.finfo(x.dtype).eps
  x[rndidx] = np.floor(x[rndidx])
  ocx = np.ceil(x).astype(int)
  evenidx = (ocx % 2) == 0
  ocx[evenidx] = ocx[evenidx] + 1
  return ocx

def odd_ceil_scalar(x):
  if np.abs( np.floor(x) - x ) <= np.finfo( type(x) ).eps:
    x = np.floor(x)
  ocx = np.ceil(x).astype(int)
  return ocx if ( ocx % 2 != 0 ) else ocx + 1

def init_map(devmin,devmax,res,cmap=None):
  if np.any(np.greater(devmin,devmax)):
    raise ValueError('[init_map]: assert(devmax >= devmin)')
  MAP = {}
  MAP['devmin']  = devmin
  MAP['devmax']  = devmax
  MAP['res'] = res
  MAP['origin'] = (devmax+devmin)/2.0 # meters
  MAP['size'] = odd_ceil_array( (devmax-devmin) / res ) # always odd
  MAP['dev'] = MAP['size'] * MAP['res'] / 2.0
  MAP['min'] = MAP['origin'] - MAP['dev']
  MAP['max'] = MAP['origin'] + MAP['dev']
  MAP['map'] = cmap
  # MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
  return MAP

def meters2cells(datam, dim_min, res):
  return np.floor( (datam - dim_min)/res )

def meters2cells_cont(datam, dim_min, res):
  return (datam - dim_min)/res

def cells2meters(datac, dim_min, res):
  return (datac + 0.5)*res + dim_min

# subv2ind: np.ravel_multi_index
# ind2subv: np.unravel_index

def plot_map(ax,cmap):
  if type(ax) is AxesImage:
    # update image data
    ax.set_data(cmap)
  else:
    # setup image data for the first time
    # transpose because imshow places the first dimension on the y-axis
    h = ax.imshow( cmap.T, interpolation="none", cmap='gray_r', origin='lower', \
                   extent=(-0.5,cmap.shape[0]-0.5, -0.5, cmap.shape[1]-0.5) )
    ax.axis([-0.5, cmap.shape[0]-0.5, -0.5, cmap.shape[1]-0.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return h

def ellipsoid(xc,yc,zc,xr,yr,zr,n=20):
  '''
    [X,Y,Z]=ELLIPSOID(XC,YC,ZC,XR,YR,ZR,N) generates three
    (N+1)-by-(N+1) matrices so that SURF(X,Y,Z) produces an
    ellipsoid with center (XC,YC,ZC) and radii XR, YR, ZR.
  '''
  theta = np.linspace(-1.0,1.0,num=n+1)*np.pi
  phi = np.linspace(-1.0,1.0,num=n+1)*np.pi/2
  cosphi = np.cos(phi); cosphi[0] = 0; cosphi[n] = 0;
  sintheta = np.sin(theta); sintheta[0] = 0; sintheta[n] = 0;

  # cartesian coordinates that correspond to the spherical angles:
  X = xr * np.outer(cosphi, np.cos(theta)) + xc
  Y = yr * np.outer(cosphi, sintheta) + yc
  Z = zr * np.outer(np.sin(phi),np.ones_like(theta)) + zc
  return X, Y, Z


def ellipse_plot(A,c=None,n=20):
  '''
  %
  %  ellipse_plot(A,c,n) plots a 2D ellipse or a 3D ellipsoid 
  %  represented in the "center" form:  
  %               
  %                   (x-c)' A^{-1} (x-c) <= 1
  % 
  %  The eigenvectors of A define the principal directions of the ellipsoid 
  %  and the eigenvalues of A are the squares of the semi-axes
  %
  %  Inputs: 
  %  A: a 2x2 or 3x3 positive-definite matrix.
  %  c: a 2D or a 3D vector which represents the center of the ellipsoid.
  %  n: the number of grid points for plotting the ellipse; Default: n = 20. 
  %
  %  Output:
  %   P = dim x 127 = points representing the ellipsoid
  %   
  %   2D plot:
  %       plot(P(1,:),P(2,:));
  %   3D plot:
  %       mesh(reshape(P(1,:),(n+1),[]),
  %            reshape(P(2,:),(n+1),[]),
  %            reshape(P(3,:),(n+1),[]));
  %
  %  Nikolay Atanasov
  %  atanasov@seas.upenn.edu
  %  University of Pennsylvania
  %  15 July 2013
  %
  '''
  a = A.shape[0]
  if( a != A.shape[1] ):
    raise ValueError('Only a square matrix can represent an ellipsoid.')
  if( a > 3 ):
    raise ValueError('Cannot plot an ellipsoid with more than 3 dimensions.')
  if( a < 2 ):
    raise ValueError('Cannot plot an ellipsoid with less than 2 dimensions.')
  
  if c is None:
    c = np.zeros((a,1))
  if c.ndim == 1:
    c = c[:,None]
  
  U, S, _ = np.linalg.svd(A)
  if( a == 2 ):
    a1 = np.sqrt(S[0])
    a2 = np.sqrt(S[1])
    theta = np.linspace(0.0, 2.0*np.pi, num=n)+ 1.0/n
    P = np.einsum('ij,jk...->ik...',U.T,np.array([[a1*np.cos(theta)],[a2*np.sin(theta)]])) + c
  else:
    a1 = np.sqrt(S[0])
    a2 = np.sqrt(S[1])
    a3 = np.sqrt(S[2])
    xx,yy,zz = ellipsoid(0,0,0,a1,a2,a3,n)
    P = np.einsum('ij,jk...->ik...',U.T,np.vstack((xx.flatten(1),yy.flatten(1),zz.flatten(1)))) + c
  return P




    
    
    
