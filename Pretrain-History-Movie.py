from keras.layers import containers, AutoEncoder, Dense
from keras import models
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from utils import JPtools as jpt
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.patches as mpatches
import os

BATCH_SIZE = 555



X, xc = jpt.manyPsf(samples = 20, batch_size = BATCH_SIZE)
X_train, X_val = train_test_split(X, test_size = 0.1, random_state = 42)
xc_train, xc_val = train_test_split(X, test_size = 0.1, random_state = 42)
x_range = np.linspace(-1, 1, 40)

encoder = containers.Sequential()
encoder.add(Dense(output_dim = 40, input_dim = 20, init = 'positive_uniform', activation = 'linear'))
decoder = containers.Sequential()
decoder.add(Dense(output_dim = 20, input_dim = 40, init = 'positive_uniform', activation = 'linear'))
ae = AutoEncoder(encoder = encoder, decoder = decoder)
model = models.Sequential()
model.add(ae)
model.compile(optimizer='sgd', loss = 'mse')


fig = plt.figure()
# train, = plt.plot([], [], color = 'b', alpha = 0.1)
# val, = plt.plot([], [], color = 'r', alpha = 0.1)
red_patch = mpatches.Patch(color = 'red', label = 'Reconstruction')
blue_patch = mpatches.Patch(color = 'blue', label = 'Input Data')
## set plot characteristics



pretrain_epochs = 0
epochs_per_frame = 5
fnumber = 0

for j in xrange(0, 80, 1):
    model.fit(X_train, X_train, nb_epoch =epochs_per_frame)
    pretrain_epochs += 5
    
    model.layers[0].output_reconstruction = False
    model.compile(optimizer = 'sgd', loss = 'mse')
    
    X_pred = model.predict(X_val)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$h(x)$')
    #plt.legend(loc='lower right', handles = [red_patch, blue_patch])
    plt.xlim(-1, 1)
    plt.ylim(-0.5, 2.5)
    for i in xrange(len(X_val)):
       # plt.plot(x_range, X_val[i][0:20], color = 'b', alpha = 0.1)
        plt.plot(x_range, X_pred[i][0:40], color = 'b', alpha = 0.1)

    plt.title('Hidden Layer Output {} epochs'.format(pretrain_epochs))
    plt.savefig('movies/_out{}'.format(str(fnumber).rjust(5, '0')))
    fig.clf()
    model.layers[0].output_reconstruction = True
    model.compile(optimizer = 'sgd', loss = 'mse')
    fnumber += 1

os.system("ffmpeg -framerate 10 -pattern_type glob -i 'movies/_out*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
