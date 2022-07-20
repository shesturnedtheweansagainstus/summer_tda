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

# noisy swiss
swissn, swiss_cn = ds.make_swiss_roll(n_samples=n_points, noise=0.2, random_state=0)

# swiss roll with hole (and color map)
#swiss_hole, swiss_hole_c = ds.make_swiss_roll(n_samples=n_points, random_state=0, hole=True)

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

# klein bottle
def klein(n=30, m=40, r=10, a=20):
    theta, phi = np.meshgrid(np.linspace(0, 2.*np.pi, n), np.linspace(0, 2.*np.pi, m))
    x = (r*np.cos(theta) + a) * np.cos(phi)
    y = (r*np.cos(theta) + a) * np.sin(phi)
    z = r*np.sin(theta) * np.cos(phi/2)
    w = r*np.sin(theta) * np.sin(phi/2)
    return np.column_stack((np.concatenate(x),np.concatenate(y),np.concatenate(z),np.concatenate(w)))

klein_data = klein()+5 # (1200, 4)  MORE TESTS

# S curve
s_curve, s_curve_c = ds.make_s_curve(n_samples=n_points, random_state=0)

# noisy S curve
s_curven, s_curve_cn = ds.make_s_curve(n_samples=n_points, noise=0.1, random_state=1)

# sphere_plane
def plane(l,n):
    return np.column_stack((np.random.sample(n)*2*l-l,np.random.sample(n)*2*l-l , np.zeros(n)))
plane_pc = plane(2, 1000)
sphere_pc = tadasets.dsphere(n=1000)
sphere_plane = np.concatenate((plane_pc, sphere_pc), axis = 0)

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

def test_normal_coords(dataset, c, **kwargs):
    """
    
    """
    S = rml.Simplex()
    S.build_simplex(dataset, **kwargs)
    
    p_idx, _ = S.normal_coords()

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(", ".join([i+"="+str(kwargs[i]) for i in kwargs.keys()]) + f', dim={S.dim}')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

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

        Implement dimension estimation for noise - how sensitive ?
        added k0 another parameter see swiss roll
    
    """

    #dataset = [point, uni_point, swiss, sphere1, sphere2, sphere3, sphere4]
    #dataset = [point]
    #dataset = [sphere4]
    #dataset = [sphere2]
    dataset = [swiss, swiss_c]
    #dataset = [swissn, swiss_cn]
    #dataset = [s_curven, s_curve_cn]
    #dataset = [klein_data]
    #dataset = [s_curve, s_curve_c]
    #dataset = [sphere_plane, sphere_plane[:, 0]]

    datasets = [dataset]

    params = {'max_components':5, 'S':0.5, 'k':10, 'threshold_var':0.08, 'edge_sen':1}

    for dataset in datasets:
        #test_normal_coords_edges(dataset, dataset[:, 0], k=10, threshold_var=0.08, edge_sen=1)
        test_normal_coords(*dataset, **params)
        pass
    

    #test_3_sphere(n=3000)