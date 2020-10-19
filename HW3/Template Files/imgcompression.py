from matplotlib import pyplot as plt
import numpy as np


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N (*3 for color images)
            S: min(N, D) * 1 (* 3 for color images)
            V: D * D (* 3 for color images)
        """
        if len(X.shape) == 2:
            U, S, V = np.linalg.svd(X)
            return U, S, V
        else:
            U_r, S_r, V_r = np.linalg.svd(X[:, :, 0])
            U_g, S_g, V_g = np.linalg.svd(X[:, :, 1])
            U_b, S_b, V_b = np.linalg.svd(X[:, :, 2])
            U = np.dstack((U_r, U_g, U_b))
            S = np.dstack((S_r, S_g, S_b))
            V = np.dstack((V_r, V_g, V_b))
            return U, S, V


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if len(U.shape) == 2:
            U_k = U[:, :k]
            S_k = S[:k]
            S_k = np.reshape(S_k, (S_k.shape[0], 1))
            V_k= V[:k, :]
            print(U_k.shape)
            print(S_k.shape)
            print(V_k.shape)
            Xrebuild = np.matmul(U_k, S_k, V_k)
            return Xrebuild
        else:
            return None


    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        raise NotImplementedError

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        raise NotImplementedError