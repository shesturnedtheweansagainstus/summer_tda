from scipy.optimize         import newton
from sklearn.neighbors      import KDTree
from scipy.sparse           import csr_matrix
from scipy.sparse.csgraph   import dijkstra
from scipy.linalg           import eig
from sklearn.decomposition  import PCA
import numpy                as np
import gudhi                as gd
import warnings
import gurobipy             as gp
from gurobipy               import GRB
import matplotlib.pyplot    as plt
from kneed                  import KneeLocator
from scipy.spatial          import distance_matrix
from ripser                 import Rips
from sklearn.preprocessing  import PolynomialFeatures
import sympy                as sp
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem     import Problem
from pymoo.optimize         import minimize


warnings.filterwarnings("ignore")

def surface_to_matrix(surface, d):
    """
    Parameters
    ----------
    surface : 1-D np.array
        Expresses the coefficients of the quadratic surface
        in PolynomialFeatures convention
    
    Returns
    -------
    """
    c = surface[0]
    b = surface[1:1+d]
    A = np.zeros([d, d])
    A[np.triu_indices(d)] = surface[1+d:]
    A[np.triu_indices(d, k=1)] /= 2
    A[np.tril_indices(d, k=-1)] = A[np.triu_indices(d, k=1)]

    return A, b, c

def find_coord(A, y, alpha, dim):
    """
    
    """
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam(GRB.Param.NonConvex, 2)
    x = m.addMVar(shape=int(dim), lb=float('-inf'))

    Q = A.T @ A
    c = -2 * y.T @ A

    obj = x @ Q @ x + c @ x + y.T @ y
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstr(x@x == alpha**2, name="c")
    m.optimize()

    return x.X

def normal_coords_pca(self, **kwargs):
    """
    Computes the Riemannian normal coordinates from 
    the 'naive' algorithm.
    """
    if self.edges == None:
        return False

    n = len(self.pointcloud)
    k1 = (self.dim+2) * (self.dim+1) // 2
    self.coords = np.zeros([n, self.dim])
    computed_points = np.full(n, False)  # tracks which coordinates has been computed

    # find our base point for T_pM
    dist_matrix, predecessors = dijkstra(self.edge_matrix, return_predecessors=True)  
    p_idx = np.argmin(np.amax(dist_matrix, axis=1)) 
    p = self.pointcloud[p_idx] 
    computed_points[p_idx] = True

    # apply PCA to set up points in T_pM 
    pca = PCA(n_components=self.dim)
    edge_coords = pca.fit_transform(self.pointcloud[self.edges[p_idx]] - p)  # ! number of points should be in [d+1, 2d]
    self.coords[self.edges[p_idx]] = edge_coords
    computed_points[self.edges[p_idx]] = True

    # then interate over all other points based off of increasing distance from p
    p_dist = dist_matrix[p_idx]
    sorted_inds = np.argsort(p_dist)

    c00 = 0

    for idx in sorted_inds:
        if computed_points[idx]:
            continue
        else:
            q = self.pointcloud[idx]
            pred = predecessors[p_idx, idx]  # (index of) point before idx on the shortest path from p to idx
            b = self.pointcloud[pred]
            b_prime = self.coords[pred]

            computed_points_b = [i for i in self.edges[pred] if computed_points[i]]
            k = len(computed_points_b)
            
            # add extra points from the neighbourhood of the c_i
            if k < k1:
                extra_computed_points = [j for i in computed_points_b for j in self.edges[i] if computed_points[j] and j not in computed_points_b and j!= pred]
                extra_computed_points_idx = np.argsort(dist_matrix[idx, extra_computed_points])[:2+k1-k]  # take extra points?  argsort from pred not idx?
                computed_points_b += list(np.asarray(extra_computed_points)[extra_computed_points_idx])
                k = len(computed_points_b)
                
            computed_points_b.append(idx)

            # fit a quadratic manifold by geodesic distance to p
            computed_points_b_pca = pca.fit_transform(self.pointcloud[computed_points_b] - b)
            computed_points_b.pop()  # removes idx
            q = computed_points_b_pca[-1]  # q in PCA
            computed_points_b_pca = computed_points_b_pca[:-1]
            geodesic_b = dist_matrix[p_idx, computed_points_b]
            surface_poly = PolynomialFeatures()
            surface_matrix = surface_poly.fit_transform(computed_points_b_pca)
            surface = np.linalg.lstsq(surface_matrix, geodesic_b)[0]
            A0, b0, c0 = surface_to_matrix(surface, self.dim)
            c0 -= dist_matrix[p_idx, pred]

            # find the closest point on the manifold to q
            D0, R0 = eig(A0)
            D0 = D0.astype('float64')
            alpha0 = R0.T @ q 
            beta0 = R0.T @ b0

            t = sp.symbols('t')
            factor = np.prod((1+2*t*D0)**2)
            D1 = 1 / (1+2*t*D0)

            a_b = alpha0 - t * beta0

            #print(f'\nfactor : {factor}\n')
            #print(f'D0 : {D0}\n')
            #print(f'D1 : {D1}\n')
            #print(f'factor*(D1 * D0 * D1) : {factor*(D1 * D0 * D1)}\n')
            #print(f'factor*D1 : {factor*D1}\n')

            o = (1+2*t*D0)**2
            oo = (1+2*t*D0)
            o0 = np.diag([np.prod(np.delete(o, i)) for i in range(self.dim)]) * np.diag(D0)
            o1 = np.diag([np.prod(np.delete(o, i)) * oo[i] for i in range(self.dim)])

            poly = a_b.T @ o0 @ a_b + beta0 @ o1 @ a_b + c0 * factor
            #poly = a_b.T @ np.diag(factor*(D1 * D0 * D1)) @ a_b + beta0.T @ np.diag(factor*D1) @ a_b + c0 * factor

            poly_roots = sp.real_roots(poly, t)  # list

            a = [np.linalg.inv(np.eye(self.dim)+2*float(i)*A0) @ (q-float(i)*b0) for i in poly_roots]
            if len(a) == 0:
                c00+=1
                q = self.pointcloud[idx]
                alpha = np.linalg.norm(q-b)  # ||q-b||

                y = self.pointcloud[computed_points_b] - b  # rows are c_i-b
                y /= np.linalg.norm(y, axis=1).reshape(k, 1) * alpha
                y *= q-b
                y = np.sum(y, axis=1)  # 1-D np.array

                A = self.coords[computed_points_b] - b_prime  # (k, dim) then U (with full_matrices=False) gives (k, dim) for U and U^Tb has (dim,)
                A /= np.linalg.norm(A, axis=1).reshape(k, 1) * alpha  
                
                m = gp.Model()
                m.setParam('OutputFlag', 0)
                m.setParam(GRB.Param.NonConvex, 2)
                x = m.addMVar(shape=int(self.dim), lb=float('-inf'))

                Q = A.T @ A
                c = -2 * y.T @ A

                obj = x @ Q @ x + c @ x + y.T @ y 
                m.setObjective(obj, GRB.MINIMIZE)
                m.addConstr(x@x == alpha**2, name="c")
                m.optimize()
                self.coords[idx] = x.X + b_prime 
                computed_points[idx] = True
                
                continue

            a_dist = np.linalg.norm(np.asarray(a) - q, axis=1)
            a_idx = np.argmin(a_dist)
            a = a[a_idx]
            a_prime = b_prime + a

            alpha = np.linalg.norm(q-a)  # ||q-a||

            y = computed_points_b_pca - a  # rows are c_i-a (in PCA)
            y /= np.linalg.norm(y, axis=1).reshape(k, 1) * alpha
            y *= q-a
            y = np.sum(y, axis=1)  # 1-D np.array

            A = self.coords[computed_points_b] - a_prime  # (k, dim) then U (with full_matrices=False) gives (k, dim) for U and U^Tb has (dim,)
            A /= np.linalg.norm(A, axis=1).reshape(k, 1) * alpha  
            
            normal_coord = find_coord(A, y, alpha, self.dim)
            self.coords[idx] = normal_coord + a_prime                    
            computed_points[idx] = True
    print(c00)
    return p_idx