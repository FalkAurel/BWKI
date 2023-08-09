import numpy as np
import matplotlib.pyplot as plt

def sparseToOneHotEncoded(y, nlabel):
    """
    Returns a one-hot encoded vector for every sample. You should use it to turn this format [0, 2, 1]
    into this [1, 0, 0], [0, 0, 1], [0, 1, 0].
    """
    ausgabe = []
    for i in y:
        target = np.zeros(nlabel,)
        target[i] = 1
        ausgabe.append(target)
    return np.array(ausgabe)

def visualize(accuracy, loss, learningRate, optim):
    """
    Takes the accuracy, the loss, the learning rate and the optimizer as parameters. Returns a visualization
    of the training process.
    """
    x = [i for i in range(len(accuracy))]
    plt.style.use("seaborn")
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
    ax1.plot(x, learningRate, color="green", label = "Learning Rate")
    ax2.plot(x, loss, color="blue", label = "Loss")
    ax3.plot(x, accuracy, color="red", label = "Accuracy")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Learning Rate")
    ax2.set_title("Loss")
    ax3.set_title(f"Accuracy avg.: {sum(accuracy[-5:]) / 5 * 100}%")
    fig.canvas.manager.set_window_title(f"Optimizer: {optim.__class__.__name__}")
    plt.tight_layout()
    plt.show()
    fig.savefig('visualization.png')


def trainTestSplit(inputX, inputY, verteilung=0.8):
    """
    return trainX, trainY, testX, testY
    """
    assert len(inputX) == len(inputY), f"expected {len(inputX)} but got {len(inputY)}"
    index = int(len(inputX) * verteilung)
    trainX, testX = inputX[:index], inputX[index:]
    trainY, testY = inputY[:index], inputY[index:]
    return trainX, trainY, testX, testY

def preProcessing(data,* , method = "minMaxNormalization", newMax = 1, newMin = -1):
    """
    preProcessing gives you access to normalization and standartization techniques. It is set by default to
    minMaxNormalization with max = 1 and min = -1.
    """
    if method == "meanNormalization":
        return _meanNormalization(data)
    if method == "minMaxNormalization":
        return _minMaxNormalization(data, newMax, newMin)

def _minMaxNormalization(data, newMax, newMin):
    return (data - data.min()) / (data.max() - data.min()) * (newMax - newMin) + newMin

def _meanNormalization(data):
    mean = data.mean(axis=0)
    std = np.sqrt(np.square(data - mean).sum(axis=0) / (len(data) - 1))
    return (data - mean) / std

def shuffle(X, y):
    """
    Shuffles the dataset.
    """
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    return X[keys], y[keys]

def predictBinary(x, decisionBoundary=0.5):
    """
    This function is supposed to be used with the output of a binary classifier. It uses this decisionBoundary
    to determine whether the output should be returned as a 0 or a 1.
    """
    return np.where(x >= decisionBoundary, 1, 0)
