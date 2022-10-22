# Note: You are not allowed to import additional python packages except NumPy
import numpy as np


class ImageCompressor:
    # This class is responsible to i) learn the codebook given the training images
    # and ii) compress an input image using the learnt codebook.
    def __init__(self):
        # You can modify the init function to add / remove some fields
        self.mean_image = np.array([])
        self.principal_components = np.array([])
        self.centralized_image = np.array([])
        
    def get_codebook(self):
        # This function should return all information needed for compression
        # as a single numpy array
        
        # TODO: Modify this according to you algorithm
        mean_image_re = np.reshape(self.mean_image, (1, -1))
        principal_components_re = np.reshape(self.principal_components, (1, -1))

        codebook = np.concatenate((mean_image_re, principal_components_re), 0)
        return codebook
    
    def train(self, train_images):
        # Given a list of training images as input, this function should learn the 
        # codebook which will then be used for compression
        data_image = np.reshape(train_images, len(train_images))
        self.mean_image = np.mean(train_images, axis = 1)
        self.centralized_image = data_image - self.mean_image

        U, S, V = np.linalg.svd(self.centralized_image, full_matrices=False)
        
        self.principal_components = U
        # X_a = np.matmul(np.matmul(P, np.diag(D)), Q)
        # print(np.std(X), np.std(X_a), np.std(X - X_a))
        # ******************************* TODO: Implement this ***********************************************#
        return self.principal_components

    def compress(self, test_image):
        # Given a test image, this function should return the compressed representation of the image
        # ******************************* TODO: Implement this ***********************************************#
        test_image_compressed = test_image
        return test_image_compressed


class ImageReconstructor:
    # This class is used on the client side to reconstruct the compressed images.
    def __init__(self, codebook):
        # The codebook learnt by the ImageCompressor is passed as input when
        # initializing the ImageReconstructor
        self.mean_image = codebook[0]
        self.principal_components = codebook[1]

    def reconstruct(self, test_image_compressed):
        # Given a compressed test image, this function should reconstruct the original image
        # ******************************* TODO: Implement this ***********************************************#
        test_image_recon = test_image_compressed
        return test_image_recon