import numpy as np
import random
import cv2
import os

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, GRU
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard


if os.name == 'nt':
    import matplotlib.pyplot as plt
    from utils_video_image import plot_chart, psnr, psnr_


#Callback to print learning rate decaying over the epoches
class LearningRatePrinter(Callback):
    def init(self):
        super(LearningRatePrinter, self).init()

    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print ("learning rate -> " + str(lr))



################################################################################
################################################################################
#Main class autencoder
class Autoencoder(object):
    def __init__(this, input_shape, type):
        
        input_img = Input(shape = input_shape)

        if type==0:

            #Convolution and Maxpooling
            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(input_img)
            x = MaxPooling2D((2,2), padding='same')(x)
            
            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(x)
            x = MaxPooling2D((2,2), padding='same')(x)

            
            x = Flatten()(x)

            x = Dense(512, activation='relu', name = 'feat')(x)

            x = Dense(2048, activation='relu')(x)

            x = Reshape((16, 16, 8))(x)

            ##decoded
            
            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(x)
            x = UpSampling2D((2,2))(x)

            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(x)
            x = UpSampling2D((2,2))(x)

        if type==1:
            
            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(input_img)
            x = MaxPooling2D((2,2), padding='same')(x)
            
            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(x)
            x = MaxPooling2D((2,2), padding = 'same')(x)

            x = Flatten(  )(x)

            #test with sigmoid
            x = Dense(512, activation='relu', name = 'feat')(x)

            x = Dense(2048, activation='relu')(x)

            x = Reshape((16,16,8))(x)
            
            #decoded
            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(x)
            x = UpSampling2D((2,2))(x)

            x = Conv2D(8, (5, 5), activation='relu', border_mode='same')(x)
            x = UpSampling2D((2,2))(x)
        
        #Output
        x = Conv2D(1, (3, 3), activation='sigmoid', border_mode='same')(x)
        this.model= Model(input_img, x)
        this.model.summary()

    ############################################################################
    def evaluate(this, x_test, x_test_original, model_name, n, files, out_file):
        this.model.load_weights(model_name)
        x = this.model.predict(x_test)

        # show visual
        cdlist = random.sample( list( range( len(x_test) ) ) , n )
        
        imgs = []
        for cd in cdlist:
            ot = x[cd]
            imgs   += [ x_test[cd].reshape(64, 64), 
                        x_test_original[cd].reshape(64, 64), 
                        ot.reshape(64, 64)]

        plot_chart(imgs, n, 3, title = 'Out')

    ############################################################################
    def saveFeats(this, x_test, x_test_original, model_name):
        this.model.load_weights(model_name)
        imodel      = Model(inputs  = this.model.input,
                            outputs = this.model.get_layer('feat').output)
        
        return imodel.predict(x_test)
        
    ############################################################################
    def train(this, X_train, X_train_, X_test, X_test_, batch_size, epochs, out, dirname):
        
        #Save weights
        weights_save = ModelCheckpoint(filepath             = out, 
                                       monitor              = 'val_loss', 
                                       verbose              = 0, 
                                       save_best_only       = True, 
                                       save_weights_only    = False, 
                                       mode                 = 'min')
        
        lr_printer = LearningRatePrinter()

        #run autoencoder
        this.model.compile(loss='mean_squared_error', optimizer='adam')

        this.model.fit(X_train, X_train_, 
                  batch_size    = batch_size, 
                  nb_epoch      = epochs,
                  verbose       = 1, 
                  shuffle       = True,
                  validation_data   = (X_test, X_test_), 
                  callbacks=[TensorBoard(log_dir=dirname+'/autoencoder'),  weights_save])
 
        this.model.evaluate(X_test, X_test_, verbose=0)

#END CLASS
################################################################################
################################################################################
def plotImages(imgs, title, savefig =''):
    lines = np.floor( len(imgs) / 5 ) + 1
    plot_chart(imgs, lines, 5, title = title, savefig = savefig)



