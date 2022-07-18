import rml
import tadasets
import gudhi                as gd
import matplotlib.pyplot    as plt
import numpy                as np
import warnings
from scipy.optimize         import newton
from sklearn.neighbors      import KDTree
from scipy.sparse           import csr_matrix
from scipy.sparse.csgraph   import dijkstra
from sklearn.decomposition  import PCA
from sklearn                import datasets as ds

warnings.filterwarnings("ignore")

"""
Diagnostics for testing the naive algorithm
"""

n_points = 1500

# section of a 2-D uniform plane in R^3 with 900 points
x = np.linspace(-1, 1, 30)
y = np.linspace(-1, 1, 30)
xx,yy = np.meshgrid(x,y)
uni_point = np.array((xx.ravel(), yy.ravel())).T
uni_point = np.hstack([uni_point, np.zeros([len(uni_point), 1])])

# section of 2-D plane in R^3
pointx = np.random.uniform(size=[n_points, 1])
pointy = np.random.uniform(size=[n_points, 1])
point = np.hstack([pointx, pointy, np.zeros([len(pointx), 1])])

# swiss roll with color map
swiss, swiss_c = ds.make_swiss_roll(n_samples=n_points, random_state=0)

# swiss roll with hole (and color map)
#swiss_hole, swiss_hole_c = ds.make_swiss_roll(n_samples=n_points, hole=True, random_state=0) update?

# 2-sphere
sphere1 = tadasets.dsphere(n=n_points)

# semi-2-sphere
sphere2 = tadasets.dsphere(n=int(n_points*2))
sphere2 = sphere2[sphere2[:, 2]>=0]

# truncated semi-2-sphere
sphere3 = tadasets.dsphere(n=int(n_points*2.2))
sphere3 = sphere3[sphere3[:, 2]>=0]
sphere3 = sphere3[sphere3[:, 2]<=0.7]

# sliced truncated semi-2-sphere 
sphere4 = tadasets.dsphere(n=int(n_points*2.3))
sphere4 = sphere4[sphere4[:, 2]>=0]
sphere4 = sphere4[sphere4[:, 2]<=0.8]
cond = np.logical_and(sphere4[:, 0]>=-0.1, sphere4[:, 0]<=0.1)
cond = np.logical_and(cond, sphere4[:, 1]>=0)
sphere4 = sphere4[~cond]

def test_normal_coords_edges(dataset, c=None, k=10, threshold_var=0.08, edge_sen=1, two_d=False):
    """
    Plots with the edge point highlighted
    two_d=True for plotting 2-D dataset
    """
    S = rml.Simplex()
    S.build_simplex(dataset, k=k, threshold_var=threshold_var, edge_sen=edge_sen)
    
    p_idx, edge = S.normal_coords()

    edge_points = S.pointcloud[edge]
    normal_edge_points = S.coords[edge]

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1) if two_d else fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    if two_d:
        ax1.scatter(dataset[:,0], dataset[:, 1], c=c)  # grading by x
        ax1.scatter(edge_points[:, 0], edge_points[:, 1], color='orange')
        ax1.scatter(dataset[p_idx,0], dataset[p_idx, 1], marker='>', color='g', s=100)
        for i in range(len(dataset)):
            for k in S.edges[i]:
                ax1.plot([dataset[i][0], dataset[k][0]],[dataset[i][1], dataset[k][1]], color='black', alpha=0.1)
    else:
        ax1.scatter3D(dataset[:,0], dataset[:, 1], dataset[:, 2], c=c)  # grading by x
        ax1.scatter3D(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2], color='orange')
        ax1.scatter3D(dataset[p_idx,0], dataset[p_idx, 1], dataset[p_idx, 2], marker='>', color='g', s=100)
        for i in range(len(dataset)):
            for k in S.edges[i]:
                ax1.plot3D([dataset[i][0], dataset[k][0]],[dataset[i][1], dataset[k][1]], [dataset[i][2], dataset[k][2]], color='black', alpha=0.1)

    ax2.scatter(S.coords[:, 0], S.coords[:, 1], c=c)
    ax2.scatter(normal_edge_points[:, 0], normal_edge_points[:, 1], color='orange')

    plt.show()

def test_normal_coords(dataset, c=None, k=10, threshold_var=0.08, edge_sen=1, two_d=False):
    """
    
    """
    S = rml.Simplex()
    S.build_simplex(dataset, k=k, threshold_var=threshold_var, edge_sen=edge_sen)
    
    p_idx, _ = S.normal_coords()

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1) if two_d else fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    if two_d:
        ax1.scatter(dataset[:,0], dataset[:, 1], c=c)  # grading by x
        ax1.scatter(dataset[p_idx,0], dataset[p_idx, 1], marker='>', color='g', s=100)
        for i in range(len(dataset)):
            for k in S.edges[i]:
                ax1.plot([dataset[i][0], dataset[k][0]],[dataset[i][1], dataset[k][1]], color='black', alpha=0.1)
    else:
        ax1.scatter3D(dataset[:,0], dataset[:, 1], dataset[:, 2], c=c)  # grading by x
        ax1.scatter3D(dataset[p_idx,0], dataset[p_idx, 1], dataset[p_idx, 2], marker='>', color='g', s=100)
        for i in range(len(dataset)):
            for k in S.edges[i]:
                ax1.plot3D([dataset[i][0], dataset[k][0]],[dataset[i][1], dataset[k][1]], [dataset[i][2], dataset[k][2]], color='black', alpha=0.1)

    ax2.scatter(S.coords[:, 0], S.coords[:, 1], c=c)

    plt.show()

def test_3_sphere(n=2000, k=10, threshold_var=0.08, edge_sen=1):

    dataset = tadasets.dsphere(n=n, d=3, ambient=10)
    #dataset = dataset[dataset[:, 0]>=0]

    print(f'Number of points : {len(dataset)}')

    S = rml.Simplex()
    S.build_simplex(dataset, k=k, threshold_var=threshold_var, edge_sen=edge_sen)
    _, _ = S.normal_coords()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection = '3d')
    ax.scatter3D(S.coords[:,0], S.coords[:, 1], S.coords[:, 2], c=dataset[:, 0])  # grading by x

    plt.show()


if __name__ == '__main__':

    """
    TODO:

        Go through Ximena .ipynb

        Implement dimension estimation for noise

        Write tests for the two variants
    
    """

    #datasets = [point, uni_point, swiss, sphere1, sphere2, sphere3, sphere4]
    #datasets = [point]
    #datasets = [sphere4]
    datasets = [sphere2]
    #datasets = [swiss]

    for dataset in datasets:
        #test_normal_coords_edges(dataset, dataset[:, 0], k=10, threshold_var=0.08, edge_sen=1)
        test_normal_coords(dataset, dataset[:, 0], k=10, threshold_var=0.08, edge_sen=1)
        pass
    
    #test_3_sphere(n=3000)