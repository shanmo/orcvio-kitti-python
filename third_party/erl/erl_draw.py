
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection, PathCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection, Path3DCollection
from matplotlib.patches import Ellipse


def move_figure(fig, x, y):
  """Move figure's upper left corner to pixel (x, y)"""
  backend = mpl.get_backend()
  if backend == 'TkAgg':
    fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
  elif backend == 'WXAgg':
    fig.canvas.manager.window.SetPosition((x, y))
  else:
    # This works for QT and GTK
    # You can also use window.setGeometry
    fig.canvas.manager.window.move(x, y)


def drawPoints2D(ax, pts, **kwargs):
  '''
  pts = nx2 = points to plot using scatter
  '''
  if type(ax) is PathCollection:
    ax.set_offsets(pts[...,:2])
  else:
    h = ax.scatter(pts[...,0],pts[...,1], **kwargs)
    return h


def drawPoints3D(ax, pts, **kwargs):
  '''
  pts = nx3 = points to plot using scatter
  '''
  if type(ax) is Path3DCollection:
    #ax.set_offsets(pts[...,:3])
    ax._offsets3d = (pts[...,0],pts[...,1],pts[...,2])
  else:
    h = ax.scatter(pts[...,0],pts[...,1],pts[...,2], **kwargs)
    return h  


def drawPath2D(ax, traj):
  ''' h = drawPath2D(h,traj)
      
      traj = num_traj x num_pts x num_dim
  '''
  if(traj.ndim < 3):
    traj = traj[None,...]

  if type(ax) is LineCollection:
    ax.set_verts(traj)
  else:
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    h = ax.add_collection(LineCollection(traj,colors=colors))
    return h

def drawPath3D(ax, traj):
  ''' h = drawPath3D(h,traj)
      
      traj = num_traj x num_pts x num_dim
  ''' 
  if(traj.ndim < 3):
    traj = traj[None,...]

  if type(ax) is Line3DCollection:
    ax.set_verts(traj)
  else:
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    h = ax.add_collection(Line3DCollection(traj,colors=colors))
    return h

def drawPose2D(ax, pose, sz = 0.5):
  '''
    pose = n x 3 = [x,y,yaw]
  '''
  if(pose.ndim < 2):
    pose = pose[None,...]
    
  dt = np.array([0,2.7,-2.7])
  # xy = n x 3 x 2
  xy = np.stack((pose[...,0,None] + sz*np.cos(pose[...,2,None]+dt),\
                 pose[...,1,None] + sz*np.sin(pose[...,2,None]+dt)),axis=-1)

  if type(ax) is PolyCollection:
    ax.set_verts(xy)
  else:
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    h = ax.add_collection(PolyCollection(xy,facecolors=colors))
    return h

def drawPose3D(ax, p, R, sz = 0.5):
  '''
    Plots frames in 3D corresponding to Nx3 positions p with Nx3x3 orientations R
  '''
  if(p.ndim < 2):
    p = p[None,...]  
    R = R[None,...]

  X = np.stack((p,p + sz*R[...,0])).transpose((1,0,2))
  Y = np.stack((p,p + sz*R[...,1])).transpose((1,0,2))
  Z = np.stack((p,p + sz*R[...,2])).transpose((1,0,2))

  if isinstance(ax,list):
    h[0].set_verts(X)
    h[1].set_verts(Y)
    h[2].set_verts(Z)
  else:
    h = [ax.add_collection(Line3DCollection(X,colors=[(1.0, 0.0, 0.0)],linestyles='dashed')),\
         ax.add_collection(Line3DCollection(Y,colors=[(0.0, 1.0, 0.0)],linestyles='dashed')),\
         ax.add_collection(Line3DCollection(Z,colors=[(0.0, 0.0, 1.0)],linestyles='dashed'))]
  return h

def drawAABB2D(ax, bbox, **kwargs):
  '''
  bbox = n x 4 = n x [x1 y1 x2 y2]
  '''
  verts = np.empty(bbox.shape[:-1]+(5,2))
  verts[...,0,0], verts[...,0,1] = bbox[...,0], bbox[...,1]
  verts[...,1,0], verts[...,1,1] = bbox[...,0], bbox[...,3]
  verts[...,2,0], verts[...,2,1] = bbox[...,2], bbox[...,3]
  verts[...,3,0], verts[...,3,1] = bbox[...,2], bbox[...,1]
  verts[...,4,0], verts[...,4,1] = bbox[...,0], bbox[...,1]
  
  if(verts.ndim < 3):
    verts = verts[None,...]

  if type(ax) is LineCollection:
    ax.set_verts(verts)
  else:
    h = ax.add_collection(LineCollection(verts, **kwargs))
    return h


def drawAABB3D(ax, bbox, **kwargs):
  '''
  bbox = n x 6 = n x [x1 y1 z1 x2 y2 z2]
  '''    
  Z = [[0,1,2],[3,1,2],[3,4,2],[0,4,2],[0,1,5],[3,1,5],[3,4,5],[0,4,5]]
  cubeView = [[Z[0],Z[1],Z[2],Z[3]], # bottom
              [Z[4],Z[5],Z[6],Z[7]], # top
              [Z[0],Z[1],Z[5],Z[4]], 
              [Z[2],Z[3],Z[7],Z[6]], 
              [Z[1],Z[2],Z[6],Z[5]],
              [Z[4],Z[7],Z[3],Z[0]]] # 6 x 4 x 3
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(np.reshape(bbox[...,cubeView],(-1,4,3)))
  else:
    h = ax.add_collection(Poly3DCollection(np.reshape(bbox[...,cubeView],(-1,4,3)), **kwargs))
    return h


def drawBox2D(ax, box, **kwargs):
  '''
  box = n x 4 x 2
  '''
  if type(ax) is PolyCollection:
    ax.set_verts(box) # n x 6 x 4 x 3
  else:
    h = ax.add_collection(PolyCollection(box, **kwargs))
    return h

def drawBox3D(ax, box, **kwargs):
  '''
  box = n x 8 x 3 =
  '''
  cubeView = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[4,7,2,0]] # 6 x 4
  if type(ax) is Poly3DCollection:
    ax.set_verts(np.reshape(box[...,cubeView,:],(-1,4,3))) # n x 6 x 4 x 3
  else:
    h = ax.add_collection(Poly3DCollection(np.reshape(box[...,cubeView,:],(-1,4,3)), **kwargs))
    return h


def ellipsoid(A,c,inverse=False, num_pts=50):
  '''
  %
  %  ellipsoid(A,c) returns a set of 2D or 3D points representing
  %  ellipsoids:   
  %                   (x-c)' A (x-c) = 1
  % 
  %  The eigenvectors of A define the principal directions of the ellipsoid 
  %  and the eigenvalues of A are the inverse squares of the semi-axes
  %
  %  Inputs: 
  %  A: nx2x2 or nx3x3 positive-semidefinite matrices
  %  c: nx2 or nx3 vectors which represent the centers of the ellipsoids
  %  inverse = True means A^{-1} is provided instead of A
  %
  %  Output:
  %   P = n x num_pts x 2 or n x (num_pts/2) x num_pts x 3 = point coordinates representing the ellipsoid
  %
  %  Example:
  %    ax.plot_surface(P[0,...,0], P[0,...,1], P[0,...,2])
  %
  '''
  ndim = A.shape[-1]
  if( ndim != A.shape[-2] ):
    raise ValueError('Only a square matrix can represent an ellipsoid.')
  if( ndim > 3 ):
    raise ValueError('Cannot plot an ellipsoid with more than 3 dimensions.')
  if( ndim < 2 ):
    raise ValueError('Cannot plot an ellipsoid with less than 2 dimensions.')
  
  U, S, VH = np.linalg.svd(A)
  
  if inverse:
    radii = np.sqrt(S) # n x ndim
    VH = U.T # n x ndim x ndim
  else: 
    radii = 1.0 / np.sqrt(S)

  if( ndim == 2 ):
    theta = np.linspace(0.0, 2.0*np.pi, num=num_pts) # 50 x 1
    X = radii[...,0,None] * np.cos(theta)
    Y = radii[...,1,None] * np.sin(theta)
    P = np.stack((X,Y),axis=-1)@VH.T + c[...,None,:]
  else:
    theta = np.linspace(-np.pi,np.pi, num=num_pts)
    costheta, sintheta = np.cos(theta), np.sin(theta)
    phi = np.linspace(-np.pi/2,np.pi/2, num=int(num_pts/2))
    cosphi, sinphi = np.cos(phi), np.sin(phi)
    X = radii[...,0,None,None] * np.outer(cosphi, costheta)
    Y = radii[...,1,None,None] * np.outer(cosphi, sintheta)
    Z = radii[...,2,None,None] * np.outer(sinphi, np.ones_like(theta))
    P = np.stack((X,Y,Z),axis=-1)@VH[...,None,:,:] + c[...,None,None,:]
  return P





## OLD FUNCTIONS

#def drawEllipse(A,c=None,n=20):
#  '''
#  %
#  %  drawEllipse(A,c,n) plots a 2D ellipse or a 3D ellipsoid 
#  %  represented in the "center" form:  
#  %               
#  %                   (x-c)' A^{-1} (x-c) <= 1
#  % 
#  %  The eigenvectors of A define the principal directions of the ellipsoid 
#  %  and the eigenvalues of A are the squares of the semi-axes
#  %
#  %  Inputs: 
#  %  A: a 2x2 or 3x3 positive-definite matrix.
#  %  c: a 2D or a 3D vector which represents the center of the ellipsoid.
#  %  n: the number of grid points for plotting the ellipse; Default: n = 20. 
#  %
#  %  Output:
#  %   P = dim x 127 = points representing the ellipsoid
#  %   
#  %   2D plot:
#  %       plot(P(1,:),P(2,:));
#  %   3D plot:
#  %       mesh(reshape(P(1,:),(n+1),[]),
#  %            reshape(P(2,:),(n+1),[]),
#  %            reshape(P(3,:),(n+1),[]));
#  %
#  %  Nikolay Atanasov
#  %  atanasov@seas.upenn.edu
#  %  University of Pennsylvania
#  %  15 July 2013
#  %
#  '''
#  a = A.shape[0]
#  if( a != A.shape[1] ):
#    raise ValueError('Only a square matrix can represent an ellipsoid.')
#  if( a > 3 ):
#    raise ValueError('Cannot plot an ellipsoid with more than 3 dimensions.')
#  if( a < 2 ):
#    raise ValueError('Cannot plot an ellipsoid with less than 2 dimensions.')
#  
#  if c is None:
#    c = np.zeros((a,1))
#  if c.ndim == 1:
#    c = c[:,None]
#  
#  U, S, _ = np.linalg.svd(A)
#  if( a == 2 ):
#    a1 = np.sqrt(S[0])
#    a2 = np.sqrt(S[1])
#    theta = np.linspace(0.0, 2.0*np.pi, num=n)+ 1.0/n
#    P = np.einsum('ij,jk...->ik...',U.T,np.array([[a1*np.cos(theta)],[a2*np.sin(theta)]])) + c
#  else:
#    a1 = np.sqrt(S[0])
#    a2 = np.sqrt(S[1])
#    a3 = np.sqrt(S[2])
#    xx,yy,zz = ellipsoid(0,0,0,a1,a2,a3,n)
#    P = np.squeeze(np.einsum('ij...,jk->ik...',np.vstack((xx.flatten(1),yy.flatten(1),zz.flatten(1)))[None,...],U.T)) + c
#  return P

#def ellipsoid(xc,yc,zc,xr,yr,zr,n=20):
#  '''
#    [X,Y,Z]=ELLIPSOID(XC,YC,ZC,XR,YR,ZR,N) generates three
#    (N+1)-by-(N+1) matrices so that SURF(X,Y,Z) produces an
#    ellipsoid with center (XC,YC,ZC) and radii XR, YR, ZR.
#  '''
#  theta = np.linspace(-1.0,1.0,num=n+1)*np.pi
#  phi = np.linspace(-1.0,1.0,num=n+1)*np.pi/2
#  cosphi = np.cos(phi); cosphi[0] = 0; cosphi[n] = 0;
#  sintheta = np.sin(theta); sintheta[0] = 0; sintheta[n] = 0;

#  # cartesian coordinates that correspond to the spherical angles:
#  X = xr * np.outer(cosphi, np.cos(theta)) + xc
#  Y = yr * np.outer(cosphi, sintheta) + yc
#  Z = zr * np.outer(np.sin(phi),np.ones_like(theta)) + zc
#  return X, Y, Z
# 

#def plot_ellipse(A, center):
#    """
#    (x-c).T * A * (x-c) = 1
#    A 2x2 matrix to base the ellipse on
#    center The location of the center of the ellipse
#    note input A is in fact A inv, need to take 1./np.sqrt(D)
#    """

#    # VT is the rotation matrix that gives the orientation of the ellipsoid.
#    # https://en.wikipedia.org/wiki/Rotation_matrix
#    # http://mathworld.wolfram.com/RotationMatrix.html

#    U, D, VT = np.linalg.svd(A)

#    # radii.
#    a, b = 1. / np.sqrt(D)

#    # Major and minor semi-axis of the ellipse.
#    if a > b:
#        dx, dy = 2 * a, 2 * b
#    else:
#        dy, dx = 2 * a, 2 * b

#    ellipse = Ellipse(xy=(center[0], center[1]), width=dx, height=dy,
#                      edgecolor='g', fc='None', lw=4)

#    return ellipse


#def plot_ellipsoid(A, center, ax, color):
#    """
#    Plot the ellipsoid equation in "center form"
#    (x-c).T * A * (x-c) = 1
#    A is 3x3
#    center is 1x3
#    """

#    U, D, V = np.linalg.svd(A)
#    rx, ry, rz = 1. / np.sqrt(D)
#    u, v = np.mgrid[0:2 * np.pi:20j, -np.pi / 2:np.pi / 2:10j]

#    x = rx * np.cos(u) * np.cos(v)
#    y = ry * np.sin(u) * np.cos(v)
#    z = rz * np.sin(v)

#    E = np.dstack([x, y, z])
#    E = np.dot(E, V) + center

#    x, y, z = np.rollaxis(E, axis=-1)
#    ax.plot_surface(x, y, z, color=color, cstride=1, rstride=1, alpha=0.05)


