import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import os

from pkg_resources import resource_filename

class PlotLosses(Callback):

    def __init__(self, figure_dir='output',figure_name='loss_plot.png',figure_path='output/loss_plot.png'):
        self.figure_path = figure_path
        self.figure_name = figure_name
        self.figure_dir = figure_dir

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1


    def on_train_end(self, logs={}):

        # clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        # plt.show();
        if not os.path.isdir(self.figure_dir):
            os.makedirs(self.figure_dir)
        self.fig.savefig(os.path.join(self.figure_dir, self.figure_name))
