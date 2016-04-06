from keras.layers import containers, AutoEncoder, Dense
from keras import models
from keras.utils.visualize_util import plot
class JPAutoEncoder(object):

    '''An AutoEncoder object. Params: list of dimensions, activation function,
    initialization function.

    Typical usage:
    ae = JPAutoEncoder(dims)
    ae.pretrain(X_train, nb_epoch=100)
    ae.train(X_train, nb_epoch=100)
    ae.predict(X_val) 
    '''

    def __init__(self, dims, act = 'relu', init = 'glorot_normal'):

        '''initialize each of the pretraining models'''
        self.dims = dims
        self.act  = act
        self.init = init

        InputEncoder = containers.Sequential()
        InputEncoder.add(Dense(output_dim=dims[1], input_dim = dims[0], 
            init = 'glorot_uniform', activation = 'linear'))

        OutputDecoder = containers.Sequential()
        OutputDecoder.add(Dense(output_dim=dims[0], input_dim = dims[1],
            init = 'glorot_uniform', activation = 'linear'))

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
        #decoders.reverse()
        for enc, dec in zip(encoders, decoders):
            #print enc.get_config()['layers'][0]['output_dim']
            #print dec.get_config()['layers'][0]['input_shape']
            autoencoders.append(AutoEncoder(encoder=enc, decoder=dec, output_reconstruction = True))

        self.pretrains = []
        for ae in autoencoders:
            self.pretrains.append(models.Sequential())
            self.pretrains[-1].add(ae)
            self.pretrains[-1].compile(optimizer='sgd', loss='mse')
            
            
            

    
    def pretrain(self, X_train, nb_epoch):
        data = X_train
        for ae in self.pretrains:            
            ae.fit(data, data, nb_epoch)
            ae.layers[0].output_reconstruction = False
            ae.compile(optimizer='sgd', loss='mse')
            data = ae.predict(data)

    def fine_train(self, X_train, nb_epoch):
        weights = [ae.layers[0].get_weights() for ae in self.pretrains]

        dims = self.dims
        encoder = containers.Sequential()
        decoder = containers.Sequential()

        ## add special input encoder
        encoder.add(Dense(output_dim = dims[1], input_dim = dims[0], 
            weights = weights[0][0:2], activation = 'linear'))
        ## add the rest of the encoders
        for i in range(1, len(dims) - 1):
            encoder.add(Dense(output_dim = dims[i+1],
                weights = weights[i][0:2], activation = self.act))

        ## add the decoders from the end

        decoder.add(Dense(output_dim = dims[len(dims) - 2], input_dim = dims[len(dims) - 1],
            weights = weights[len(dims) - 2][2:4], activation = self.act))
        
        for i in range(len(dims) - 2, 1, -1):
            decoder.add(Dense(output_dim = dims[i - 1],
                weights = weights[i-1][2:4], activation = self.act))
        
        ## add the output layer decoder
        decoder.add(Dense(output_dim = dims[0], 
            weights = weights[0][2:4], activation = 'linear'))

        plot(encoder, to_file = 'encoder.png', show_shape = True)
        
        masterAE = AutoEncoder(encoder = encoder, decoder = decoder)
        masterModel = models.Sequential()
        masterModel.add(masterAE)
        masterModel.compile(optimizer = 'sgd', loss = 'mse')
        masterModel.fit(X_train, X_train, nb_epoch)
        self.model = masterModel

    def predict(self, X_test):
        return self.model.predict(X_test)


        
