from sklearn.neighbors      import KDTree
from sklearn.decomposition  import PCA
import numpy                as np
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
        edges : (n_edges,) np.array
            The ith entry contains the indexes of the 'safe' points which 
            connect to the ith point.

        Attributes for Testing
        --------------------
        vis : (n_edges,) np.array
            Same as edges but for 'visible' points.
        dims : (n_samples,) list
            Collection of 1-D np.arrays which track how
            PCA estimates the intincit dimension about each point
        vars : (n_samples,) list
            Collection of 1-D np.arrays which track how
            PCA estimates the intrinsic dimension about each point
        """
        self.pointcloud = None
        self.edges = None

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
        which connect to the 'idx' point.

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
        """
        point = self.pointcloud[idx]
        edges = self.pointcloud[ind] - point  # ascending by length
        dims = []
        vars = []
        threshold_edge = edge_sen * np.mean(dist)

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
                return ind[:j-1], dims, vars
            
            dim0 = dim1

        return ind, dims, vars

    def build_edges(self, pointcloud, k=10, threshold_var=0.02, edge_sen=0.5):
        """
        Computes the edges of our simplex.

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
        kd_tree = KDTree(pointcloud, leaf_size=2)  # ALREADY ORDERED ASCENDING
        dists, inds = kd_tree.query(pointcloud, k=k+1)
        dists = dists[:, 1:]  # removes points being compared to itself with KNN
        inds = inds[:, 1:]

        visible_edges = [self.find_visible_edge(i, inds[i], dists[i]) for i in range(n)]
        self.vis = visible_edges
        safe_edges = [self.find_safe_edges(i, visible_edges[i][0], visible_edges[i][1], threshold_var, edge_sen) for i in range(n)]
        self.edges = [safe_edges[i][0] for i in range(n)]

        # for testing
        self.dims = [np.asarray(safe_edges[i][1]) for i in range(n)]
        self.vars = [safe_edges[i][2] for i in range(n)]

    def find_dim(self):
        """
        Computes the dimension of our pointcloud from
        its simplex.
        """
        pass