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
from scipy.spatial.distance import directed_hausdorff
from sklearn.decomposition  import PCA
from sklearn                import datasets as ds
from pathlib import Path


warnings.filterwarnings("ignore")

"""
Diagnostics for testing the algorithm with noise
"""

def test_noise(dataset, noise_dataset, c, cn, save=False, **kwargs):
    """
    
    """

    S = rml.Simplex()
    S.build_simplex(dataset, **kwargs)

    Sn = rml.Simplex()
    Sn.build_simplex(noise_dataset, **kwargs)
    
    p_idx, _ = S.normal_coords()
    p_idxn, _ = Sn.normal_coords()

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'k={kwargs["k"]}, threshold_var={kwargs["threshold_var"]}, edge_sen={kwargs["edge_sen"]}')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.scatter3D(dataset[:,0], dataset[:, 1], dataset[:, 2], c=c)
    ax1.scatter3D(dataset[p_idx,0], dataset[p_idx, 1], dataset[p_idx, 2], marker='>', color='g', s=100)
    for i in range(len(dataset)):
        for k in S.edges[i]:
            ax1.plot3D([dataset[i][0], dataset[k][0]],[dataset[i][1], dataset[k][1]], [dataset[i][2], dataset[k][2]], color='black', alpha=0.1)

    ax2.scatter(S.coords[:, 0], S.coords[:, 1], c=c)

    ax3.scatter3D(noise_dataset[:,0], noise_dataset[:, 1], noise_dataset[:, 2], c=cn)
    ax3.scatter3D(noise_dataset[p_idxn,0], noise_dataset[p_idxn, 1], noise_dataset[p_idxn, 2], marker='>', color='g', s=100)
    for i in range(len(noise_dataset)):
        for k in Sn.edges[i]:
            ax3.plot3D([noise_dataset[i][0], noise_dataset[k][0]],[noise_dataset[i][1], noise_dataset[k][1]], [noise_dataset[i][2], noise_dataset[k][2]], color='black', alpha=0.1)
    
    ax4.scatter(Sn.coords[:, 0], Sn.coords[:, 1], c=cn)

    if save:
            folder = Path("/home/lz1919/Documents/UNI/year_three/summer_tda/pic")
            save_name = save + '.png'
            fig.savefig(folder / save_name)

    plt.show()

def compare_normal_coords(dataset, c, save=False, **kwargs):
    """
    
    """

    S0 = rml.Simplex()
    S0.build_simplex(dataset, **kwargs)

    S1 = rml.Simplex()
    S1.build_simplex(dataset, **kwargs)
    
    p_idx, _ = S0.normal_coords()
    p_idx, _ = S1.normal_coords_new()

    fig = plt.figure(figsize=(30, 30))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.scatter3D(dataset[:,0], dataset[:, 1], dataset[:, 2], c=c)
    ax1.scatter3D(dataset[p_idx,0], dataset[p_idx, 1], dataset[p_idx, 2], marker='>', color='g', s=100)
    for i in range(len(dataset)):
        for k in S0.edges[i]:
            ax1.plot3D([dataset[i][0], dataset[k][0]],[dataset[i][1], dataset[k][1]], [dataset[i][2], dataset[k][2]], color='black', alpha=0.1)

    ax2.scatter(S0.coords[:, 0], S0.coords[:, 1], c=c)
    ax3.scatter(S1.coords[:, 0], S1.coords[:, 1], c=c)

    haus_dist = directed_hausdorff(S0.coords, S1.coords)
    fig.suptitle(f'k={kwargs["k"]}, threshold_var={kwargs["threshold_var"]}, edge_sen={kwargs["edge_sen"]}\nHausdorff Distance={haus_dist}')

    if save:
            folder = Path("/home/lz1919/Documents/UNI/year_three/summer_tda/pic")
            save_name = save + '.png'
            fig.savefig(folder / save_name)

    plt.show()

if __name__ == '__main__':

    """
    TODO:

        Implement variants 1, 2 from Ximena
        Dimensionality estimation
        Edge detection

    """

    n = 1000
    
    params = {'k':10, 'threshold_var':0.08, 'edge_sen':0.5}

    # 2-plane
    pointx = np.random.uniform(size=[n, 1])
    pointy = np.random.uniform(size=[n, 1])
    point = np.hstack([pointx, pointy, np.zeros([len(pointx), 1])])
    pointn = point + np.random.normal(scale=0.05, size=point.shape)
    point_c = point[:, 0]
    point_cn = pointn[:, 0]
    point_data = [point, pointn, point_c, point_cn]

    # swiss roll
    swiss, swiss_c = ds.make_swiss_roll(n_samples=n, random_state=0)
    swissn, swiss_cn = ds.make_swiss_roll(n_samples=n, noise=0.2, random_state=0)
    swiss_data = [swiss, swissn, swiss_c, swiss_cn]  # edge_sen:0.8 ?

    # semi-2-sphere
    sphere = tadasets.dsphere(n=n, d=2)
    sphere = sphere[sphere[:, 2]>=0]
    spheren = tadasets.dsphere(n=n, d=2, noise=0.05)
    spheren = spheren[spheren[:, 2]>=0]
    sphere_data = [sphere, spheren, sphere[:, 0], spheren[:, 0]]

    # 2-torus
    torus = tadasets.torus(n=n)
    torusn = tadasets.torus(n=n, noise=0.2)
    torus_c = torus[:, 0]
    torus_cn = torusn[:, 0]
    torus_data = [torus, torusn, torus_c, torus_cn]

    # s-curve
    s_curve, s_curve_c = ds.make_s_curve(n_samples=n, random_state=0)
    s_curven, s_curve_cn = ds.make_s_curve(n_samples=n, noise=0.2, random_state=0)
    s_curve_data = [s_curve, s_curven, s_curve_c, s_curve_cn]

    #test_noise(*point_data, save='point_data_1', **params)
    #test_noise(*s_curve_data, **params)
    compare_normal_coords(s_curve, s_curve_c, save=False, **params)