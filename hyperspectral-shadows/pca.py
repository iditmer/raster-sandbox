import numpy as np

class principal_components:

    def __init__(self, data_matrix: np.ndarray):

        if len(data_matrix.shape) != 2:
            raise ValueError('Array passed to principal component constructor has unexpected dimensions. Should be 2D matrix.')
        
        # center data to make SVD consistent with the eigendecomposition of the covariance of data_matrix;
        # store the mean for proper dimensionality reduction and reconstruction operations
        self.data_mean = np.mean(data_matrix, axis=0)
        centered_data = data_matrix - self.data_mean

        # S will contain singular values in descending order, V will contain right-eigenvectors of [ A^T A ]
        U, S, V = np.linalg.svd(centered_data, full_matrices=False)

        # singular values are related to eignvalues of covariance as s^2 / (n - 1) when using Bessel's correction for covariance
        # scaling these values to their sum provides a %-age of variance explained by each principal component
        eigenvalues = (S ** 2) / (data_matrix.shape[0] - 1)
        self.variance_explained = eigenvalues / np.sum(eigenvalues)

        # numpy svd returns V as a collection of row-eigenvectors; store as column vectors 
        self.principal_components = V.T

    def rotate_to_principal_axes(self, data_matrix: np.ndarray) -> np.ndarray:
        
        if len(data_matrix.shape) != 2:
            raise ValueError('Array passed to principal component transformation has unexpected dimensions. Should be 2D matrix.')
        
        return np.matmul(data_matrix, self.principal_components)

    def reduce_dimension(self, data_matrix: np.ndarray, number_components: int) -> np.ndarray:
        
        if len(data_matrix.shape) != 2:
            raise ValueError('Array passed to principal component transformation has unexpected dimensions. Should be 2D matrix.')
        
        if number_components < 1 or number_components > len(self.variance_explained):
            raise ValueError('Number of components specified in dimensionality reduction is not valid for this PCA instance.')

        # data_matrix is n x p, reduce dimensionality through multiplying with p x k matrix of first k column eigenvcetors of length p
        return np.matmul(data_matrix - self.data_mean, self.principal_components[:, :number_components])
    
    def reconstruct(self, data_matrix: np.ndarray) -> np.ndarray:
        
        if len(data_matrix.shape) != 2:
            raise ValueError('Array passed to principal component transformation has unexpected dimensions. Should be 2D matrix.')
        
        if data_matrix.shape[1] > len(self.variance_explained):
            raise ValueError('Data matrix passed to reconstruction has greater dimensionality than expected for this PCA instance.')
        
        # data_matrix is n x k, restore to original dimensionality through left-multiplication of transpose by p x k eigenvector matrix
        return (np.matmul(self.principal_components[:, :data_matrix.shape[1]], data_matrix.T).T + self.data_mean)