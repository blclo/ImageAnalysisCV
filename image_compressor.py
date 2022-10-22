import numpy as np


class ImageCompressor:
    # This class is responsible to i) learn the codebook given the training images
    # and ii) compress an input image using the learnt codebook.
    def __init__(self):
        self.mean_image = np.array([])
        self.principal_components = np.array([])
      
    def get_codebook(self):
        # This function should return all information needed for compression
        # as a single numpy array
        mean_image_re = np.reshape(self.mean_image, (1, -1)) # 1 x n_pixels(96*96*3)
        principal_components_re = (self.principal_components.T) # Nxd (n_images x n_pixels(96*96*3))
        codebook = np.concatenate((mean_image_re, principal_components_re), 0) # on top of each other
        return codebook
        
    def train(self, train_images):
        # Given a list of training images as input, this function should learn the 
        # codebook which will then be used for compression
        
        self.mean_image =(sum(train_images)/len(train_images))
        print(" The mean image is")
        print(self.mean_image)
        # calculate principal_componants
        
        X = (np.reshape(train_images,(len(train_images),-1)))
        X_meaned = X - np.reshape(self.mean_image,(1,-1))
        
        U,S,V = np.linalg.svd(X_meaned.T, full_matrices=False)
        
        self.principal_components = U[:,:15]
        
        return

    def compress(self, test_image):
        # Given a test image, this function should return the compressed representation of the image
        # ******************************* TODO: Implement this ***********************************************#
        U = self.principal_components
        mean = np.reshape(self.mean_image,(-1,1))
        test_image = np.reshape(test_image,(-1,1))
        test_image_compressed = (U.T).dot(test_image-mean) #n_priceple_comp x 1
        return test_image_compressed


class ImageReconstructor:
    # This class is used on the client side to reconstruct the compressed images.
    def __init__(self, codebook):
        # The codebook learnt by the ImageCompressor is passed as input when
        # initializing the ImageReconstructors
        self.mean_image = np.reshape(codebook[0],(96,96,3)).astype(np.float64)
        self.principal_components = (codebook[1:,:]).T.astype(np.float64)
    def reconstruct(self, test_image_compressed):
        # Given a compressed test image, this function should reconstruct the original image
        # ******************************* TODO: Implement this ***********************************************#
        U = self.principal_components
        mean = self.mean_image
        X_rec = np.reshape(U.dot(test_image_compressed),(96,96,3)) + mean
        
        #X_rec[X_rec>160] = 255
        #X_rec[X_rec<250] = 0
        tresh = 200
        tresh2 = 150
        tresh3 = 150
        tresh4 = 100

        white = np.logical_and(X_rec[:,:,0]>tresh,np.logical_and(X_rec[:,:,1]>tresh,X_rec[:,:,2]>tresh)) 
        green = np.logical_and(X_rec[:,:,0]<tresh2,np.logical_and(X_rec[:,:,1]>tresh2,X_rec[:,:,2]<tresh2))
        red = np.logical_and(X_rec[:,:,0]>tresh3,np.logical_and(X_rec[:,:,1]<tresh3,X_rec[:,:,2]<tresh3)) 
        black = np.logical_and(X_rec[:,:,0]<tresh4,np.logical_and(X_rec[:,:,1]<tresh4,X_rec[:,:,2]<tresh4))

        X_rec[white,:] = [255,255,255]
        X_rec[green,:] = [0,255,0]
        X_rec[red,:] = [255, 0,0]
        X_rec[black,:] = [0, 0,0]
        #X_rec[X_rec<=np.array([100,100,100])] = 0
        #X_rec[X_rec[:,:,1]>=0,0] = 255
        return (X_rec)
