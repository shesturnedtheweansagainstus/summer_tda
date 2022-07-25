from traceback import print_tb
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

warnings.filterwarnings("ignore")

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

def local_pca_elbow(pointcloud, max_components, S):
    """
    Applies PCA to local pointclouds and recover local dimension finding elbows in the function of recovered variances
    """

    if len(pointcloud) == 1:
        return 1
    
    #pca = PCA(n_components=max_components)
    pca = PCA()
    _ = pca.fit(pointcloud)
    vs = pca.explained_variance_ratio_
    
    kneedle = KneeLocator([i for i in range(len(vs))], vs, S=S, curve='convex', direction='decreasing')
    elbow = kneedle.elbow
    dim = elbow + 1 if elbow!=None else 0
    
    return dim

def local_neighborhood(data, scale):
    '''
    Computes an annular neighborhood for every point in the point cloud.
    
    data: array describing the point cloud
    scale: list [k1, k2] with k1, k2 integers describing the number of nearest neighbors that will comprise the annulus
    '''
    D = distance_matrix(data, data)
    n = scale[1]-scale[0]
    local_neigh = np.ndarray(shape=(len(D),n), dtype=int, order='F')
    radius = np.ndarray(shape=(len(D),2), dtype=float, order='F')
    for i in range(len(D)):
        local_neigh[i] = np.argsort(D[i])[scale[0]:scale[1]] # the annulus neighborhood of point i in data
    D.sort()
    for i in range(len(D)):
        radius[i] = [D[i][scale[0]], D[i][scale[1]]] # the pair [r1,r2] of radii associated to the annulus neighborhood
    return local_neigh, radius 

def compute_local_persistence(data, scale, d):
    '''
    Classify every point in the point cloud depending its local homology at degree d-1
    
    data: array describing the point cloud
    scale: list [k1, k2] with k1, k2 integers describing the number of nearest neighbors that will comprise the annulus
    d: the estimated intrinsic dimension of the point cloud
    '''
    k1 = scale[0]
    k2 = scale[1]
    
    local_neigh, radius = local_neighborhood(data, [k1, k2])
    
    rips = Rips(maxdim = d-1)
    mask = []
    for i in range(len(data)):
        dgm = rips.fit_transform(data[local_neigh[i]])
        
        # here we only focus on betti d-1

        lifetime = dgm[d-1][:,1]-dgm[d-1][:,0]

        r1 = radius[i][0]
        r2 = radius[i][1]
            
        N = np.where(lifetime>r2-r1)[0]

        if len(N)==0:
            mask.append(0) # boundary
        elif len(N)==1:
            mask.append(1) # regular point
        else:
            mask.append(2) # singular point

    return np.array(mask)

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
        

class Simplex:
    """
    Represents the simplex skeleton of 
    our pointcloud.
    """

    def __init__(self):
        """
        Attributes
        ----------
        pointcloud : (n_samples, n_features) np.array
            The pointcloud data from which we build our simplex. 
            (TODO be able to add points?)
        Simplex : gd.SimplexTree
            Stores the simplex structure with GUDHI.
        edges : (n_samples,) list containing 1-D np.array
            The ith entry contains the indexes of the 'safe' points which 
            connect to the ith point.
        edge_matrix : (n_samples, n_samples) np.array // csr_matrix
            Gives the structure of our edge connection.
        dim : int
            The dimension of our simplex.
        coords : (n_samples, self.dim) np.array
            Riemannian normal coordinates from the 'naive' algorithm.
        Attributes for Testing
        --------------------
        vis : (n_edges,) np.array
            Same as edges but for 'visible' points.
        dims : (n_samples,) list
            Collection of 1-D np.arrays which track how
            PCA estimates the intincit dimension about each point.
        vars : (n_samples,) list
            Collection of 1-D np.arrays which track how
            PCA estimates the intrinsic dimension about each point.
        """
        self.pointcloud = None
        self.simplex = gd.SimplexTree()
        self.edges = []
        self.edge_matrix = None
        self.dim = None  # check other methods for dim estimation
        self.coords = None

        self.vis = None
        self.dims = None
        self.vars = None
    
    def find_visible_edge(self, idx, ind, dist):
        """
        Computes a list of the indexes of points visible from 
        the 'idx' point, and their distances from this point 
        (in ascending length).
        Parameters
        ----------
        idx : int
            Index of a point.
        ind : (k,) np.array
            Indexes of points connected to the idx point by KNN.
        dist : (k,) np.array
            Array of edge lengths from KNN.
        
        Returns
        -------
        visible_ind : (n_visible_edges,) np.array
            Indexes of visible edges (self.pointcloud).
        visible_dist : (n_visible_edges,) np.array
            Lengths of visible edges.
        """
        point = self.pointcloud[idx]
        # List of indexes for visible points from the 'idx' point
        # where the indexes are for 'ind' (not self.pointcloud)
        visible_points_idx = [] 
        visible = True

        for y_count, idy in enumerate(ind):
            y = self.pointcloud[idy]
            for idz in ind:
                if idz != idy:
                    z = self.pointcloud[idz]
                    cos_angle = np.dot(point - z, y - z)  
                    if cos_angle < 0:
                        visible = False
                        break
            if visible == True:
                visible_points_idx.append(y_count)
            visible = True
        
        visible_dist = dist[visible_points_idx]  
        visible_ind = ind[visible_points_idx] 

        return visible_ind, visible_dist 
    
    def find_safe_edges(self, idx, ind, dist, threshold_var, edge_sen):
        """
        Computes the list of safe edges of points from visible edges 
        which connect to the 'idx' point and stores in our SimplexTree.
        Parameters
        ----------
        idx : int
            Index of a point.
        ind : (k,) np.array
            Indexes of visible points connected to the idx point.
        dist : (k,) np.array
            Array of edge lengths.
        threshold_var : [0,1] float
            The threshold to estimate the local intrinsic dimension by PCA.
        edge_sen : positive float
            The sensitivity with which we choose safe edges.
        Returns
        -------
        ind : 1-D np.array
            Indexes of safe points connected to the 'idx' point.
        TODO add rest
        """
        point = self.pointcloud[idx]
        edges = self.pointcloud[ind] - point  # ascending by length
        threshold_edge = edge_sen * np.mean(dist)
        self.simplex.insert([idx, ind[0]])  # first point is always included

        # for testing
        dims = []
        vars = []
        
        for j in range(2, len(edges)+1):  # need len != 1 
            pca = PCA()
            pca.fit_transform(edges[:j])
            var = pca.explained_variance_ratio_

            vars.append(var)
            dims.append(np.sum(var >= threshold_var))

            if j==2:
                dim0 = dim1 = np.sum(var >= threshold_var)
            else:
                dim1 = np.sum(var >= threshold_var)

            if dim1>dim0 and dist[j-1]-dist[j-2]>threshold_edge:
                #self.edges.append(ind[:j-1])
                self.edge_matrix[idx, ind[:j-1]] = dist[:j-1]
                self.edge_matrix[ind[:j-1], idx] = dist[:j-1]
                return dims, vars
            
            dim0 = dim1

            self.simplex.insert([idx, ind[j-1]])

        #self.edges.append(ind)
        self.edge_matrix[idx, ind] = dist
        self.edge_matrix[ind, idx] = dist

        return dims, vars

    def build_simplex(self, pointcloud, max_components=5, S=0.1, k=10, threshold_var=0.08, edge_sen=1, **kwargs):
        """
        Computes the edges of our simplex and the GUDHI simplex tree.
        Parameters
        ----------
        pointcloud : (n_samples, n_features) np.array
            The pointcloud data from which we build our simplex.
        k : int
            The number of NN we use.
        threshold_var : [0,1] float
            The threshold to estimate the local intrinsic dimension by PCA.
        edge_sen : positive float
            The sensitivity with which we choose safe edges.
        """
        n = len(pointcloud)
        self.pointcloud = pointcloud
        self.edge_matrix = np.zeros([n, n])

        kd_tree = KDTree(pointcloud, leaf_size=2)  # ALREADY ORDERED ASCENDING
        dists, inds = kd_tree.query(pointcloud, k=k+1)
        dists = dists[:, 1:]  # removes points being compared to itself with KNN
        inds = inds[:, 1:]

        visible_edges = [self.find_visible_edge(i, inds[i], dists[i]) for i in range(n)]
        self.vis = visible_edges
        dims_vars = [self.find_safe_edges(i, visible_edges[i][0], visible_edges[i][1], threshold_var, edge_sen) for i in range(n)]
        self.edges = [np.where(self.edge_matrix[i]!=0)[0] for i in range(n)]  # ensures can see all edges to i

        max_components = min(max_components, len(pointcloud[0]))
        local_dims = [local_pca_elbow(pointcloud[edges], max_components, S) for edges in self.edges]
        self.dim = np.max(local_dims)  # check

        self.edge_matrix = csr_matrix(self.edge_matrix)

        #self.simplex.expansion(1000)  # likely max needed
        #self.dim = self.simplex.dimension()

        # for testing (TODO: convert with logging)
        self.dims = [np.asarray(dims_vars[i][0]) for i in range(n)]
        self.vars = [dims_vars[i][1] for i in range(n)]

    def old_normal_coords(self, **kwargs):
        """
        Computes the Riemannian normal coordinates from 
        the 'naive' algorithm.
        """
        if self.edges == None:
            return False
                
        n = len(self.pointcloud)
        self.coords = np.zeros([n, self.dim])
        computed_points = np.full(n, False)  # tracks which coordinates has been computed

        edge = np.full(n, False)  # tracks edge points

        # find our base point for T_pM
        dist_matrix, predecessors = dijkstra(self.edge_matrix, return_predecessors=True) 
        p_idx = np.argmin(np.amax(dist_matrix, axis=1))  # assumes connected
        p = self.pointcloud[p_idx] 
        computed_points[p_idx] = True

        # set up tangent basis
        tangent_inds = np.random.choice(self.edges[p_idx], size=self.dim, replace=False)
        tangent_edges = np.transpose(self.pointcloud[tangent_inds] - p)  # problem if dim=1??  (dim, dim)
        tangent_edges = np.linalg.qr(tangent_edges)[0]  # gives orthonormal basis for T_pM
        
        # compute normal coords for p's edge points
        edge_points = np.transpose(self.pointcloud[self.edges[p_idx]] - p)
        edge_scalar = np.linalg.norm(edge_points, axis=0)
        edge_coords = np.linalg.lstsq(tangent_edges, edge_points)[0]
        edge_coords = (edge_coords / np.linalg.norm(edge_coords, axis=0)) * edge_scalar
        self.coords[self.edges[p_idx]] = np.transpose(edge_coords)
        computed_points[self.edges[p_idx]] = True

        edge[self.edges[p_idx]] = True

        # then interate over all other points based off of increasing distance from p??
        p_dist = dist_matrix[p_idx]
        sorted_inds = np.argsort(p_dist)

        for idx in sorted_inds:
            if computed_points[idx]:
                continue
            else:
                q = self.pointcloud[idx]
                pred = predecessors[p_idx, idx]  # (index of) point before idx on the shortest path from p to idx ! not -9999
                computed_points_b = [i for i in self.edges[pred] if computed_points[i]]

                # we add the indexes of computed points connected to the c_i which are not already in the list and are not b
                computed_points_b += [j for i in computed_points_b for j in self.edges[i] if computed_points[j] and j not in computed_points_b and j!= pred]
                    
                k = len(computed_points_b)

                b = self.pointcloud[pred]
                b_prime = self.coords[pred]

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

        return p_idx, edge

    def normal_coords(self, k0=0, **kwargs):
        """
        Computes the Riemannian normal coordinates from 
        the 'naive' algorithm.
        """
        if self.edges == None:
            return False
                
        n = len(self.pointcloud)
        self.coords = np.zeros([n, self.dim])
        computed_points = np.full(n, False)  # tracks which coordinates has been computed

        edge = np.full(n, False)  # tracks edge points

        # find our base point for T_pM
        dist_matrix, predecessors = dijkstra(self.edge_matrix, return_predecessors=True) 
        p_idx = np.argmin(np.amax(dist_matrix, axis=1))  # assumes connected
        p = self.pointcloud[p_idx] 
        computed_points[p_idx] = True

        # set up tangent basis
        tangent_inds = np.random.choice(self.edges[p_idx], size=self.dim, replace=False)
        tangent_edges = np.transpose(self.pointcloud[tangent_inds] - p)  # problem if dim=1??  (dim, dim)
        tangent_edges = np.linalg.qr(tangent_edges)[0]  # gives orthonormal basis for T_pM
        
        # compute normal coords for p's edge points
        edge_points = np.transpose(self.pointcloud[self.edges[p_idx]] - p)
        edge_scalar = np.linalg.norm(edge_points, axis=0)
        edge_coords = np.linalg.lstsq(tangent_edges, edge_points)[0]
        edge_coords = (edge_coords / np.linalg.norm(edge_coords, axis=0)) * edge_scalar
        self.coords[self.edges[p_idx]] = np.transpose(edge_coords)
        computed_points[self.edges[p_idx]] = True

        edge[self.edges[p_idx]] = True

        # then interate over all other points based off of increasing distance from p??
        p_dist = dist_matrix[p_idx]
        sorted_inds = np.argsort(p_dist)

        for idx in sorted_inds:
            if computed_points[idx]:
                continue
            else:
                q = self.pointcloud[idx]
                pred = predecessors[p_idx, idx]  # (index of) point before idx on the shortest path from p to idx ! not -9999

                computed_points_b_c = [i for i in self.edges[pred] if computed_points[i]]

                # we add the indexes of computed points connected to the c_i which are not already in the list and are not b

                if len(computed_points_b_c) < self.dim+k0:  # TODO change how many points we take?
                    extra_computed_points = [j for i in computed_points_b_c for j in self.edges[i] if computed_points[j] and j not in computed_points_b_c and j!= pred]
                    extra_computed_points_idx = np.argsort(dist_matrix[idx, extra_computed_points])
                    computed_points_b_d = list(np.asarray(extra_computed_points)[extra_computed_points_idx[:k0+self.dim-len(computed_points_b_c)]])

                computed_points_b = computed_points_b_c + computed_points_b_d
                k = len(computed_points_b)

                b = self.pointcloud[pred]
                b_prime = self.coords[pred]

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

        return p_idx, edge

    def new_normal_coords(self, k0=0, beta=1, **kwargs):
        """
        Computes the Riemannian normal coordinates from 
        the 'naive' algorithm.
        """
        if self.edges == None:
            return False
                
        n = len(self.pointcloud)
        self.coords = np.zeros([n, self.dim])
        computed_points = np.full(n, False)  # tracks which coordinates has been computed

        edge = np.full(n, False)  # tracks edge points

        # find our base point for T_pM
        dist_matrix, predecessors = dijkstra(self.edge_matrix, return_predecessors=True) 
        p_idx = np.argmin(np.amax(dist_matrix, axis=1))  # assumes connected
        p = self.pointcloud[p_idx] 
        computed_points[p_idx] = True

        # set up tangent basis
        tangent_inds = np.random.choice(self.edges[p_idx], size=self.dim, replace=False)
        tangent_edges = np.transpose(self.pointcloud[tangent_inds] - p)  # problem if dim=1??  (dim, dim)
        tangent_edges = np.linalg.qr(tangent_edges)[0]  # gives orthonormal basis for T_pM
        
        # compute normal coords for p's edge points
        edge_points = np.transpose(self.pointcloud[self.edges[p_idx]] - p)
        edge_scalar = np.linalg.norm(edge_points, axis=0)
        edge_coords = np.linalg.lstsq(tangent_edges, edge_points)[0]
        edge_coords = (edge_coords / np.linalg.norm(edge_coords, axis=0)) * edge_scalar
        self.coords[self.edges[p_idx]] = np.transpose(edge_coords)
        computed_points[self.edges[p_idx]] = True

        edge[self.edges[p_idx]] = True

        # then interate over all other points based off of increasing distance from p??
        p_dist = dist_matrix[p_idx]
        sorted_inds = np.argsort(p_dist)

        for idx in sorted_inds:
            if computed_points[idx]:
                continue
            else:
                q = self.pointcloud[idx]
                pred = predecessors[p_idx, idx]  # (index of) point before idx on the shortest path from p to idx ! not -9999

                computed_points_b_c = [i for i in self.edges[pred] if computed_points[i]]

                # we add the indexes of computed points connected to the c_i which are not already in the list and are not b

                if len(computed_points_b_c) < self.dim+k0:  # TODO change how many points we take?
                    extra_computed_points = [j for i in computed_points_b_c for j in self.edges[i] if computed_points[j] and j not in computed_points_b_c and j!= pred]
                    extra_computed_points_idx = np.argsort(dist_matrix[idx, extra_computed_points])
                    computed_points_b_d = list(np.asarray(extra_computed_points)[extra_computed_points_idx[:k0+self.dim-len(computed_points_b_c)]])

                computed_points_b = computed_points_b_c + computed_points_b_d
                k = len(computed_points_b)

                b = self.pointcloud[pred]
                b_prime = self.coords[pred]

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

                obj1 = x @ Q @ x + c @ x + y.T @ y 

                #obj2 = 0
                #for i in computed_points_b_c:
                dist_1 = dist_matrix[idx, computed_points_b_c[0]]
                c_prime_1 = self.coords[computed_points_b_c[0]]
                d_1 = b_prime - c_prime_1

                if len(computed_points_b_c) == 2:
                    dist_2 = dist_matrix[idx, computed_points_b_c[1]]
                    c_prime_2 = self.coords[computed_points_b_c[1]]
                    d_2 = b_prime - c_prime_2

                    a = (x@x + 2 * d_1 @ x + d_1@d_1 - dist_1**2) ** 2
                    b = (x@x + 2 * d_2 @ x + d_2@d_2 - dist_2**2) ** 2
                    obj2 = a + b 
                else:
                    a = (x@x + 2 * d_1 @ x + d_1@d_1 - dist_1**2) ** 2
                    obj2 = a # minimise distance of q' to c_i'

                obj = beta*obj1 + (1-beta)*obj2
                m.setObjective(obj, GRB.MINIMIZE)
                m.addConstr(x@x == alpha**2, name="c")
                m.optimize()
                self.coords[idx] = x.X + b_prime                    
                        
                computed_points[idx] = True

        return p_idx, edge

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

    def compute_boundary(self):
        """
        
        """

        mask = compute_local_persistence(self.coords, [40, 80], self.dim)  # test more parameters
        dist_matrix, predecessors = dijkstra(self.edge_matrix, return_predecessors=True) 
        boundary_points = np.where(mask==0)[0]
        p_idx = boundary_points[0]
        p_dist = dist_matrix[p_idx, boundary_points]
        return boundary_points, p_dist


def old_compute_boundary(S0, **kwargs):
    """
    
    """
    mask = compute_local_persistence(S0.coords, [40, 80], S0.dim)  # test more parameters
    dist_matrix, predecessors = dijkstra(S0.edge_matrix, return_predecessors=True) 
    boundary_points = np.where(mask==0)[0]

    p_idx = boundary_points[0]
    p_dist = dist_matrix[p_idx, boundary_points]

    S_b = Simplex()
    S_b.build_simplex(S0.coords[boundary_points], **kwargs)

    b_idxs = []

    for i, edges in enumerate(S_b.edges):
        i_weight = np.sum(S_b.edge_matrix[i])  # normalize?
        j_weights = [np.sum(S_b.edge_matrix[j]) for j in edges]
        if np.all(i_weight > np.asarray(j_weights)):
            continue
        else:
            b_idxs.append(i)

    S_c = Simplex()
    S_c.build_simplex(S_b.pointcloud[b_idxs], **kwargs)

    return boundary_points, p_dist, S_c

def compute_boundary(S0, **kwargs):
    """
    
    """
    mask = compute_local_persistence(S0.coords, [40, 80], S0.dim)  # test more parameters
    dist_matrix, predecessors = dijkstra(S0.edge_matrix, return_predecessors=True) 
    boundary_points = np.where(mask==0)[0]

    p_idx = boundary_points[0]
    p_dist = dist_matrix[p_idx, boundary_points]

    S_b = Simplex()
    S_b.build_simplex(S0.coords, **kwargs)

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(S_b.pointcloud[:, 0], S_b.pointcloud[:, 1])
    ax1.scatter(S_b.pointcloud[boundary_points, 0], S_b.pointcloud[boundary_points, 1], color='r')

    for i in range(len(S_b.pointcloud)):
        for k in S_b.edges[i]:
            ax1.plot([S_b.pointcloud[i][0], S_b.pointcloud[k][0]],[S_b.pointcloud[i][1], S_b.pointcloud[k][1]], color='black', alpha=0.1)

    b_idxs = []

    for i, edges in enumerate(np.asarray(S_b.edges)[boundary_points]):
        if np.argmin(S_b.edge_matrix[i, edges]) in boundary_points:
            b_idxs.append(i)
        else:
            continue
    
    S_c = Simplex()
    S_c.build_simplex(S0.coords[boundary_points[b_idxs]], **kwargs)

    return boundary_points, p_dist, S_c     

if __name__ == '__main__':
    pass

