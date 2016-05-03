import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from utils import JPtools as jpt
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.patches as mpatches
import os
#NETWORK_DIMS = [20, 40, 30, 15, 7, 3]
#NETWORK_DIMS = [20, 15, 10, 5, 4, 3]
#NETWORK_DIMS = [20, 10, 7, 5]
NETWORK_DIMS = [20, 40, 20, 15, 7, 3]

#NETWORK_DIMS = [10, 40, 10, 7, 3]
BATCH_SIZE = 555
X_BOUND = 1
XC_BOUND = 0.4
SAMPLES = 20
SIGMA = 0.2

matplotlib.use("Agg")
jae = jpt.JPAutoEncoder(NETWORK_DIMS, act = 'relu', init = 'glorot_normal',
    outer_init = 'positive_uniform')

X, xc = jpt.manyPsf(samples = SAMPLES, batch_size = BATCH_SIZE, x_bound = X_BOUND, xc_bound = XC_BOUND, sigma = SIGMA)
X_train, X_val = train_test_split(X, test_size = 0.1, random_state = 42)
xc_train, xc_val = train_test_split(X, test_size = 0.1, random_state = 42)
x_range = np.linspace(-X_BOUND, X_BOUND, SAMPLES)
jae.pretrain(X_train, num_epoch = 200)


fig = plt.figure()
red_patch = mpatches.Patch(color = 'red', label = 'Reconstruction')
blue_patch = mpatches.Patch(color = 'blue', label = 'Input Data')


full_network_epochs = 0
epochs_per_frame = 20
fnumber = 0

for j in xrange(0, 250, 1):
    jae.fine_train(X_train, epochs_per_frame)
    full_network_epochs += 20
    X_pred = jae.predict(X_val)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$I(x)$')
    plt.legend(loc='lower right', handles = [red_patch, blue_patch])
    plt.xlim(-1, 1)
    plt.ylim(-1, 4)
    for i in xrange(len(X_val)):
        plt.plot(x_range, X_val[i][0:SAMPLES], color = 'b', alpha = 0.1)
        plt.plot(x_range, X_pred[i][0:SAMPLES], color = 'r', alpha = 0.1)

    plt.title('{} epochs'.format(full_network_epochs))
    plt.savefig('movie_temp/_out{}'.format(str(fnumber).rjust(5, '0')))
    fig.clf()
    fnumber += 1


## make the movie
os.system("ffmpeg -framerate 10 -pattern_type glob -i \
    'movie_temp/_out*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
