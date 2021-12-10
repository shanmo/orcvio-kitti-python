import numpy as np 

def uline_zb_to_uline_b(T, S, P, uline_zb):
    '''
     * @Input:
     *    T = 4 x 4 = object pose, i.e., transform from object to world frame
     *    S = 4 x 4 = inverse camera pose, i.e., transform from world to optical frame
     *    P = 3 x 4 = projection matrix 
     *    uline_zb = 3 x 1 = 2D line in homogeneous coordinates 
     * @Output:
     *    uline_b = 4 x 1  
    '''

    uline_b = T.T @ S.T @ P.T @ uline_zb

    return uline_b

def batch_outer_product(a, b):
    """
    return a @ b.T
    >>> a = np.random.rand(2, 3)
    >>> b = np.random.rand(2, 2)
    >>> np.allclose(batch_outer_product(a, b)[0], a[0, :, None] @ b[0, None, :])
    True
    """
    return (a[..., None] * b[..., None, :])

def batch_matmul(A, B):
    """
    >>> A = np.random.rand(1, 3, 2)
    >>> B = np.random.rand(1, 2, 1)
    >>> np.allclose(batch_matmul(A, B), A @ B)
    True
    """
    return (A[..., None] * B[..., None, :, :]).sum(-2)

def batch_vecmat_mul(b, A):
    """
    return b.T @ A
    >>> A = np.random.rand(2, 3, 2)
    >>> b = np.random.rand(2, 3)
    >>> np.allclose(batch_vecmat_mul(b, A)[0], b[0].T @ A[0])
    True
    """
    return batch_matmul(b[..., None, :], A).squeeze(-2)

def batch_dot_prod(a, b):
    """
    >>> a = np.random.rand(2, 3)
    >>> b = np.random.rand(2, 3)
    >>> np.allclose(batch_dot_prod(b, a)[0], b[0].T @ a[0])
    True
    """
    return (a * b).sum(-1)

def batch_quadratic_form(b, A, c):
    """
    return b.T @ A @ c
    >>> A = np.random.rand(2, 3, 2)
    >>> b = np.random.rand(2, 3)
    >>> c = np.random.rand(2, 2)
    >>> np.allclose(batch_quadratic_form(b, A, c)[0], b[0].T @ A[0] @ c[0])
    True
    """
    return (b[..., :, None] * A * c[..., None, :]).sum(-1).sum(-1)

def batch_norm(X):
    """
    >>> x = np.random.rand(3)
    >>> np.allclose(batch_norm(x), np.linalg.norm(x))
    True
    """
    return np.sqrt((X*X).sum(axis=-1))

class normalize_up:
    def __new__(cls, up):
        """
        up: 4x1 equation of plane in homogeneous coordinates
        returns up / |up[:-1]|_2
        >>> up = np.array([0, 3, 4, 5])
        >>> norm_up = normalize_up(up)
        >>> np.allclose(norm_up, np.array([0, 0.6, 0.8, 1]))
        True
        >>> up = np.random.rand(4)
        >>> norm_up = normalize_up(up)
        >>> np.allclose(batch_norm(norm_up[..., :-1]), 1)
        True
        >>> up = np.random.rand(10, 4)
        >>> norm_up = normalize_up(up)
        >>> np.allclose(batch_norm(norm_up[..., :-1]), 1)
        True
        """
        return up / batch_norm(up[..., :-1])[..., None]


    @classmethod
    def df(cls, up):
        """
        f = up / |up|₂
                ∂ f(up)
        returns —————–   = I4x4/|up|₂ - up @ up[:-1].T @ [I₃ₓ₃ 0] / |up|³
                ∂ up
        >>> up = np.random.rand(4)
        >>> anaJ = normalize_up.df(up)
        >>> numJ = numerical_jacobian(normalize_up, up)
        >>> np.allclose(anaJ, numJ, atol=1e-3)
        True
        >>> UP = np.random.rand(10, 4)
        >>> anaJ = normalize_up.df(UP)
        >>> numJ = numerical_jacobian(normalize_up, UP)
        >>> np.allclose(anaJ, numJ, atol=1e-3)
        True
        """
        D = up.shape[-1]
        pnorm = batch_norm(up[..., :-1])[..., None, None]
        P = np.zeros((D - 1, D))
        P[:, :-1] = np.eye(D-1)
        return np.eye(D) / pnorm - batch_outer_product(up, batch_vecmat_mul(up[..., :-1], P)) / pnorm**3


def signum(X):
    return np.where(X > 0, 1, np.where(X < 0, -1, 0))


class ellipse_plane_dist_full:
    def __new__(cls, x0, Usq, hat_up):
        """
        D = 3 for 3D
        D = 2 for 2D
        x0     : (D+1,) center in homogeneous coordinates
        Usq   : DxD dual ellipsoid in euclidean coordinates
        hat_up : (D+1,) equation of plane in homogeneous coordinates already normalized
        returns the shortest distance of the ellipsoid (x0, Usq) from the hyperplane up
        >>> x0 = np.array([0, 0, 0, 1])
        >>> Usq = np.diag([1, 4, 9])
        >>> up = np.random.rand(4)
        >>> hat_up = up / batch_norm(up[:-1])
        >>> sqrt_pUsqp = np.sqrt(batch_quadratic_form(hat_up[:-1], Usq, hat_up[:-1]))
        >>> d1 = ellipse_plane_dist_full(x0, Usq, hat_up)

        Batch version
        >>> x0 = np.array([0, 0, 0, 1])
        >>> Usq = np.diag([1, 4, 9])
        >>> UP = np.random.rand(10, 4)
        >>> hat_UP = normalize_up(UP)
        >>> sqrt_pUsqp = np.sqrt(batch_quadratic_form(hat_UP[..., :-1], Usq, hat_UP[..., :-1]))
        >>> d1s = ellipse_plane_dist_full(x0, Usq, hat_UP)
        """
        plane_orig_dist = batch_dot_prod(x0, hat_up)
        sign = signum(plane_orig_dist)
        return plane_orig_dist - sign * get_sqrt_bU2b(Usq, hat_up)


    @classmethod
    def df(cls, x0, Usq, hat_up):
        """
        D = 3 for 3D, 2 for 2D
        x0   : (D+1,) center in homogeneous coordinates
        Usq : DxD dual ellipsoid in euclidean coordinates
        hat_up   : (D+1,) equation of plane in normalized homogeneous coordinates
        returns the shortest distance of the ellipsoid (x0, Usq) from the hyperplane up
        >>> up = np.random.rand(4)
        >>> x0 = np.array([0, 0, 0, 1])
        >>> Usq = np.diag([1, 2, 3])
        >>> hat_up = normalize_up(up)
        >>> numJ = numerical_jacobian(
        ...        lambda X: ellipse_plane_dist_full(x0, Usq, X)[..., None], hat_up)
        >>> anaJ = ellipse_plane_dist_full.df(x0, Usq, hat_up)
        >>> np.allclose(numJ, anaJ, atol=1e-3)
        True

        Batch version
        >>> UP = np.random.rand(10, 4)
        >>> x0 = np.array([0, 0, 0, 1])
        >>> Usq = np.diag([1, 2, 3])
        >>> hat_UP = normalize_up(UP)
        >>> numJ = numerical_jacobian(
        ...        lambda X: ellipse_plane_dist_full( x0, Usq, X)[..., None], hat_UP)
        >>> anaJ = ellipse_plane_dist_full.df(x0, Usq, hat_UP)
        >>> np.allclose(numJ, anaJ, atol=1e-3)
        True
        """
        D = x0.shape[-1] - 1
        plane_orig_dist = batch_dot_prod(x0, hat_up)
        sign = signum(plane_orig_dist)[..., None, None]

        PQP = np.zeros((D+1, D+1))
        PQP[:-1, :-1] = Usq
        return x0[..., None, :] - sign * batch_matmul(hat_up[..., None, :], PQP) / get_sqrt_bU2b(Usq, hat_up)[..., None, None]



def numerical_jacobian(f, X, d=1e-6):
    """
    >>> x = np.random.rand(3)
    >>> f = np.sin
    >>> np.allclose(numerical_jacobian(f, x), np.diag(np.cos(x)), atol=1e-3)
    True
    >>> x = np.random.rand(3)
    >>> A = np.array([[2, 3, 11], [5, 7, 13]])
    >>> f = lambda X: batch_matmul(A, X[..., None]).squeeze(-1)
    >>> np.allclose(numerical_jacobian(f, x), A)
    True

    Batch version
    >>> x = np.random.rand(10, 3)
    >>> f = np.sin
    >>> np.allclose(numerical_jacobian(f, x)[0, ...], np.diag(np.cos(x[0])), atol=1e-3)
    True
    >>> x = np.random.rand(10, 3)
    >>> A = np.array([[2, 3, 11], [5, 7, 13]])
    >>> f = lambda X: batch_matmul(A, X[..., None]).squeeze(-1)
    >>> np.allclose(numerical_jacobian(f, x), A)
    True
    """
    dX = np.eye(X.shape[-1]) * d
    XpDx = X[..., None, :] + dX
    Jacobian_transpose = (f(XpDx) - f(X[..., None, :]))/d
    return np.swapaxes(Jacobian_transpose, -1, -2)

def get_sqrt_bU2b(Usq, hat_up):
    """
    compute \sqrt{\hat{\mathbf{b}}^{\top} \mathbf{U}^{2} \hat{\mathbf{b}}} 
    """

    hat_p = hat_up[..., :-1]
    # bQb = batch_quadratic_form(hat_p, Usq, hat_p)
    bQb = batch_quadratic_form(hat_up[..., :-1], Usq, hat_up[..., :-1])
    return np.sqrt(bQb)




