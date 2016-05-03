## Author: Elias Rubin
## Dependencies: Keras (Tensorflow)
## Special note: 'positive_uniform' initialization requires modifying keras
## source, in keras/initializations.py. (Trivial exentension from uniform).

## note: if running in an ipython notebook, use %%capture cell magic for the
## pretraining and fine training functions.


from keras.layers import containers, AutoEncoder, Dense
from keras import models
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

def makePsf(x, sigma1 = 1.0, b= 0.0, sigma_ratio = 2, xc = 0):    
    I = np.exp(-0.5*((x - xc)/sigma1)**2) + b*np.exp(-0.5*((x - xc)/(sigma_ratio*sigma1))**2)
    I /= np.sum(I)*(x[1] - x[0])

    return I


def manyPsf(samples = 20, batch_size = 5000, sigma = 0.2, x_bound = 1, xc_bound = 0.4):
    np.random.seed(seed=42)
    X = []
    xcList = np.random.uniform(-xc_bound, xc_bound, batch_size)

    for xc in xcList:
        x0 = np.zeros(samples)
        x0[0:samples] = makePsf(np.linspace(-x_bound, x_bound, samples), sigma1 = sigma, xc = xc)
        X.append(x0)
    X = np.vstack(X)
    return X, xcList

def chi2(x_val, rec):
    '''x_val: np.array(dims)
       rec: np.array(dims)
       Compute the chi^2 value corresponding to these inputs.'''
    return np.sum(np.square(rec - x_val))

def nCount(xc, xc_train, epsilon = 0.01):
    '''xc: floating point
       xc_train: numpy array (number of training samples)
       Count the number of values in xc_train within [xc - epsilon, xc + epsilon]'''
    count = 0
    for i in xc_train:
        if i > (xc - epsilon) and i < (xc + epsilon):
            count += 1
    return count

class JPAutoEncoder(object):

    '''An AutoEncoder object. Params: list of dimensions, activation function,
    initialization function.

    Typical usage:
    ae = JPAutoEncoder(dims)
    ae.pretrain(X_train, nb_epoch=100)
    ae.train(X_train, nb_epoch=100)
    ae.predict(X_val) 
    '''

    def __init__(self, dims, act = 'relu', input_act = 'linear', output_act = 'linear', init = 'glorot_normal', outer_init = 'glorot_uniform'):

        '''initialize each of the pretraining models'''
        self.dims = dims
        self.act  = act
        self.input_act = input_act
        self.output_act = output_act
        self.init = init

        ## the input encoder is special because it may have a different initializtion
        ## and activation function
        InputEncoder = containers.Sequential()
        InputEncoder.add(Dense(output_dim=dims[1], input_dim = dims[0], 
            init = outer_init, activation = input_act))

        ## ditto with the output decoder
        OutputDecoder = containers.Sequential()
        OutputDecoder.add(Dense(output_dim=dims[0], input_dim = dims[1],
            init = outer_init, activation = output_act))

        ## this implementation assumes each encoder/decoder is treated the same
        hiddenEncoders = [containers.Sequential() for i,j in zip(dims[1:], dims[2:])]
        for encoder, sizes, in zip(hiddenEncoders, zip(dims[1:], dims[2:])):
            encoder.add(Dense(output_dim=sizes[1], input_dim = sizes[0],
                init = init, activation = act))


        hiddenDecoders = [containers.Sequential() for i,j in zip(dims[1:], dims[2:])]
        for decoder, sizes in zip(hiddenDecoders, zip(dims[1:], dims[2:])):
            decoder.add(Dense(output_dim=sizes[0], input_dim=sizes[1],
                init = init, activation = act))


        encoders = []
        encoders.append(InputEncoder)
        for i in hiddenEncoders:
            encoders.append(i)

        decoders = []
        decoders.append(OutputDecoder)
        for i in hiddenDecoders:
            decoders.append(i)
        

        autoencoders = []
        
        for enc, dec in zip(encoders, decoders):
            autoencoders.append(AutoEncoder(encoder=enc, decoder=dec, 
                output_reconstruction = True))

        self.pretrains = []
        for ae in autoencoders:
            self.pretrains.append(models.Sequential())
            self.pretrains[-1].add(ae)
            self.pretrains[-1].compile(optimizer='sgd', loss='mse')
            
            
    def pretrain(self, X_train, num_epoch = 25, **kwargs):
        data = X_train
        for ae in self.pretrains:            
            ae.fit(data, data, nb_epoch = num_epoch, **kwargs)
            ae.layers[0].output_reconstruction = False
            ae.compile(optimizer='sgd', loss='mse')
            data = ae.predict(data)

    def fine_train(self, X_train, num_epoch = 25, **kwargs):
        if hasattr(self, 'model'):
            return self._continue_training(X_train, num_epoch, **kwargs)

        weights = [ae.layers[0].get_weights() for ae in self.pretrains]

        dims = self.dims
        encoder = containers.Sequential()
        decoder = containers.Sequential()

        ## add special input encoder
        encoder.add(Dense(output_dim = dims[1], input_dim = dims[0], 
            weights = weights[0][0:2], activation = self.input_act))
        ## add the rest of the encoders
        for i in xrange(1, len(dims) - 1):
            encoder.add(Dense(output_dim = dims[i+1],
                weights = weights[i][0:2], activation = self.act))

        ## add the decoders from the end

        decoder.add(Dense(output_dim = dims[len(dims) - 2], input_dim = dims[len(dims) - 1],
            weights = weights[len(dims) - 2][2:4], activation = self.act))
        
        for i in xrange(len(dims) - 2, 1, -1):
            decoder.add(Dense(output_dim = dims[i - 1],
                weights = weights[i-1][2:4], activation = self.act))
        
        ## add the output layer decoder
        decoder.add(Dense(output_dim = dims[0], 
            weights = weights[0][2:4], activation = self.output_act))

        #plot(encoder, to_file = 'encoder.png', show_shape = True)
        
        masterAE = AutoEncoder(encoder = encoder, decoder = decoder)
        masterModel = models.Sequential()
        masterModel.add(masterAE)
        masterModel.compile(optimizer = 'sgd', loss = 'mse')
        masterModel.fit(X_train, X_train, nb_epoch = num_epoch, **kwargs)
        self.model = masterModel

    def _continue_training(self, X_train, num_epoch, **kwargs):
        self.model.fit(X_train, X_train, nb_epoch = num_epoch, **kwargs)

    def predict(self, X_test):
        if not hasattr(self, 'model'):
            raise AttributeError('You need to train before you predict')
        return self.model.predict(X_test)

    def predict_and_show(self, X_test, num_epoch, num_samples, savefig = True):
        if not hasattr(self, 'model'):
            raise AttributeError('You need to train before you predict')

        x_range = np.linspace(-1, 1, len(X_test[0]))
        n = len(X_test[0])
        f = plt.figure()
        f.set_size_inches(8, 6)
        for i in xrange(len(X_test)):
            plt.plot(x_range, X_test[i][0:n], color = 'b', alpha = 0.1)
            plt.plot(x_range, self.model.predict(X_test)[i][0:n], color = 'r', alpha = 0.1)
        plt.xlabel(r"$x$", size=14)
        plt.ylabel(r"$I(x)$", size=14)
        plt.title("Predictions and True Values for Validation Set {} epochs, {} samples".format(num_epoch, num_samples))
        red_patch = mpatches.Patch(color = 'red', label = 'Reconstruction')
        blue_patch = mpatches.Patch(color = 'blue', label = 'Input Data')
        plt.legend(loc='lower right', handles = [red_patch, blue_patch])
        
        if savefig:
            plt.savefig("plots/Validation-{}-epochs-{}-samples.pdf".format(num_epoch, num_samples))
        
        plt.show()

    def validation_chi2(self, X_val):
        ''' return the spatial average validation chi2 for the current model '''

        X_pred = self.predict(X_val)
        chi2s = [chi2(X_val[i], X_pred[i]) for i in range(X_val.shape[0])]
        return np.mean(chi2s)



    def chi2_plot(self, xc_val, X_val, num_epoch, num_samples, savefig = True):
        if not hasattr(self, 'model'):
            raise AttributeError('You need to train before you predict')

        X_pred = self.model.predict(X_val)

        chi2s = [chi2(X_val[i], X_pred[i]) for i in range(len(xc_val))]

        f = plt.figure()
        f.set_size_inches(8, 6)
        plt.scatter(xc_val, chi2s, alpha = 0.3)
        plt.xlabel(r"$x_c$", size = 14)
        plt.ylabel(r"$\chi^2$", size = 14)
        plt.title(r"$\chi^2$ Validation Set {} epochs, {} samples".format(num_epoch, num_samples))
        if savefig:
            plt.savefig("plots/chi2-{}-epochs-{}-samples.pdf".format(num_epoch, num_samples))

        plt.show()

    def chi2_counts_plot(self, xc_train, num_epoch, num_samples, epsilon = 0.02, sigma = 0.2, savefig = True):
        if not hasattr(self, 'model'):
            raise AttributeError('You need to train before you predict')

        xc_inspect = np.arange(-2*sigma, 2*sigma, epsilon)
        xc_train_counts = {i : nCount(i, xc_train, epsilon) for i in xc_inspect}
        chi2_xc = {}
        x_space = np.linspace(-1, 1, 20)
        for i in xc_inspect:
            val_sample = np.vstack((makePsf(x_space, sigma1 = 0.2, xc = i), 
                makePsf(x_space, sigma1 = 0.2, xc = i)))
            val_sample_pred = self.predict(val_sample)
            chi2_sample = chi2(val_sample[0], val_sample_pred[0])
            chi2_xc[i] = chi2_sample

        f = plt.figure()
        f.set_size_inches(8, 6)

        for k in xc_train_counts.keys():
            plt.scatter(xc_train_counts[k], chi2_xc[k], c = 
                (np.exp(-np.abs(k)/15), np.exp(-np.abs(k)), np.exp(-np.abs(k)/15)))

        plt.xlabel('Counts in training set', size = 14)
        plt.ylabel(r'$\chi^2$ for validation set', size = 14)
        plt.title(r'$x_c$ from {} to {}, with bins of size {}, {} epochs, and {} training samples'.format(
            -2*sigma, 2*sigma, epsilon, num_epoch, num_samples))
        if savefig:
            plt.savefig('plots/Counts-Chi2-{}-epochs-{}-samples.pdf'.format(num_epoch, num_samples))










        
