from dataclasses import replace
from sklearn.neighbors      import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition  import PCA
import numpy                as np
import gudhi                as gd
import warnings
warnings.filterwarnings("ignore")

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
        self.dim = None
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
                self.edges.append(ind[:j-1])
                self.edge_matrix[idx, ind[:j-1]] = dist[:j-1]
                self.edge_matrix[ind[:j-1], idx] = dist[:j-1]
                return dims, vars
            
            dim0 = dim1

            self.simplex.insert([idx, ind[j-1]])

        self.edges.append(ind)
        self.edge_matrix[idx, ind] = dist
        self.edge_matrix[ind, idx] = dist

        return dims, vars

    def build_simplex(self, pointcloud, k=10, threshold_var=0.02, edge_sen=0.5):
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
        self.edge_matrix = csr_matrix(self.edge_matrix)

        self.simplex.expansion(1000)  # likely max needed
        self.dim = self.simplex.dimension()

        # for testing (TODO: convert with logging)
        self.dims = [np.asarray(dims_vars[i][0]) for i in range(n)]
        self.vars = [dims_vars[i][1] for i in range(n)]

    def normal_coords(self):
        """
        Computes the Riemannian normal coordinates from 
        the 'naive' algorithm.
        """
        if self.edges == None:
            return False
        n = len(self.pointcloud)
        self.coords = np.zeros([n, self.dim])

        computed_points = {i: False for i in range(n)}  # tracks which coordinates has been computed

        # find our base point for T_pM
        dist_matrix, predecessors = dijkstra(self.edge_matrix, return_predecessors=True)  
        p_idx = np.argmin(np.amax(dist_matrix, axis=1))  # assumes connected
        p = self.pointcloud[p_idx]
        self.coords[p_idx] = 0  

        # set up tangent basis
        tangent_inds = np.random.choice(self.edges[p_idx], size=self.dim, replace=False)
        tangent_edges = np.transpose(self.pointcloud[tangent_inds] - p)  # problem if dim=1??
        
        # compute normal coords for p's edge points
        edge_points = np.transpose(self.pointcloud[self.edges[p_idx]] - p)
        edge_scalar = np.linalg.norm(edge_points, axis=0)
        edge_coords = np.linalg.lstsq(tangent_edges, edge_points)[0]
        edge_coords = (edge_coords / np.linalg.norm(edge_coords, axis=0)) * edge_scalar
        self.coords[self.edges[p_idx]] = np.transpose(edge_coords)

        




