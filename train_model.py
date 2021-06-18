from glob import glob
import numpy as np
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

"""
class plot_acc_val(Callback):
    # def on_epoch_end(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("End epoch {} of training; got log keys: {}".format(epoch, keys))
    def __init__(self, name):
        super().__init__()
        self.name = 'acc_loss'
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        plt.ion()

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        self.plot_figure(epoch, logs)

    def plot_figure(self, epoch, logs):
        self.logs.append(logs)
        self.losses.append(logs.get("loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_losses.append(logs.get("val_loss"))
        self.val_acc.append(logs.get("val_accuracy"))
        print("[Epoch{}]".format(epoch + 1))
        # Before plotting ensure at least 2 epochs have passed
        # if len(self.losses) > 1:
        n = np.arange(0, len(self.losses))
        # You can chose the style of your preference
        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure(1)
        plt.clf()
        plt.subplot(211)
        plt.plot(n, self.acc, label="train_acc")
        plt.plot(n, self.val_acc, label="val_acc")
        plt.title("Training Accuracy [Epoch {}]".format(epoch + 1))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.legend()

        plt.subplot(212)
        plt.plot(n, self.losses, label="train_loss")
        plt.plot(n, self.val_losses, label="val_loss")
        plt.title("Training LOSS [Epoch {}]".format(epoch + 1))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.legend()
        # Make sure there exists a folder called output in the current directory
        # or replace "output" with whatever directory you want to put in the plots
        plt.show()
        plt.pause(0.01)
        plt.savefig("./train_weight/" + self.name + ".png")
        pass
"""

class ConfusionMatrix(Callback):
    def __init__(self, x_val, y_val, classes, normalize=True, name='confusion_matrix'):
        super().__init__()
        self.name = name
        self.x_val = x_val
        self.y_val = y_val
        self.classes = classes
        self.normalize = normalize
        self.val_acc = []
        self.count = 1

    def on_epoch_end(self, epoch, logs=None):
        self.val_acc.append(logs.get("val_accuracy"))
        if len(self.val_acc) >= 2:
            if self.val_acc[-1] > self.val_acc[-2]:
                pred = self.model.predict(self.x_val)
                y_pred = np.argmax(pred, axis=1)
                y_true = np.argmax(self.y_val, axis=1)
                [tn, fp], [fn, tp] = confusion_matrix(y_true, y_pred)
                self.classes = ['Vigorous', 'fatigue']
                self.plot_confusion_matrix(cnf_matrix=[[tn, fp], [fn, tp]],
                                           classes=self.classes, epoch=epoch, normalize=True,
                                           name=self.name
                                           )

    def on_train_end(self, logs={}):
        self.val_acc = []
        self.count += 1
        plt.close()

    def plot_confusion_matrix(self, cnf_matrix, classes, epoch, name, normalize=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cnf_matrix = np.array(cnf_matrix)

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        else:
            print('Confusion matrix, without normalization')
        plt.figure(2)
        plt.clf()
        plt.imshow(cnf_matrix, interpolation='nearest', cmap="Blues")
        plt.title(name + ":[Epoch {}]".format(epoch))
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = np.max(cnf_matrix) / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            print(i,j,cnf_matrix[i, j])
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('./train_weight/confusion_matrix/' + name + "_" + str(self.count) + '.png')



def min_max_normalize(input_array):
    # x: input image data in numpy array
    # return normalized x
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    output = (input_array - min_val) / (max_val - min_val)  # array
    return output
