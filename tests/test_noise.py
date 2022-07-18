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
Diagnostics for testing the algorithm with noise
"""

def test_noise(dataset, noise_dataset, c, cn, **kwargs):
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

    plt.show()

if __name__ == '__main__':

    """
    TODO:

        Implement variants 1, 2 from Ximena
        Dimensionality estimation
        Edge detection

    """

    n = 1000
    
    params = {'k':10, 'threshold_var':0.08, 'edge_sen':1}

    swiss, swiss_c = ds.make_swiss_roll(n_samples=n, random_state=0)
    swissn, swiss_cn = ds.make_swiss_roll(n_samples=n, noise=0.2, random_state=0)
    swiss_data = [swiss, swissn, swiss_c, swiss_cn]  # edge_sen:0.8 ?

    sphere = tadasets.dsphere(n=n, d=2)
    sphere = sphere[sphere[:, 2]>=0]
    spheren = tadasets.dsphere(n=n, d=2, noise=0.05)
    spheren = spheren[spheren[:, 2]>=0]
    sphere_data = [sphere, spheren, sphere[:, 0], spheren[:, 0]]

    test_noise(*sphere_data, **params)