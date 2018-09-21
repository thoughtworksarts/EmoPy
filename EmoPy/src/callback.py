from matplotlib import pyplot as plt
from keras.callbacks import Callback

class PlotLosses(Callback):

    def __init__(self, figure_path='output/loss_plot.png'):
        self.figure_path = figure_path

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
        self.fig.savefig(self.figure_path)