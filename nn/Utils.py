import numpy as np
import matplotlib.pyplot as plt

def sparseToOneHotEncoded(y, nlabel):
    """
    Gibt einen One-Hot-Encoded-Vector zurück. Wenn die Labels in einem Format wie diesem Vorliegen:
    [1, 2, 0] sind sie für diese Libary nicht zu gebrauchen.
    """
    ausgabe = []
    for i in y:
        target = np.zeros(nlabel,)
        target[i] = 1
        ausgabe.append(target)
    return np.array(ausgabe)

def visualize(accuracy, loss, learningRate, optim):
    """
    Gibt eine Visualisierung des neuronalen Netzes aus.
    Nimmt die Genauigkeit, den Loss, die LernRate und den Optimizer als args. Die ersten 3 müssen als
    Liste übergeben werden.
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

def preProcessing(data):
    norm = (data.max() - data.min()) / 2
    return (data - norm) / norm

def shuffle(X, y):
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    return X[keys], y[keys]

def predictBinary(x, decisionBoundary=0.5):
    """
    Nimmt den Output des Netzes.
    Gibt einen Array von der shape x.shape mit entweder 0 oder 1 zurück.
    Sollte als decision-boundary für binary Classifier genutzt werden.
    """
    return np.where(x >= decisionBoundary, 1, 0)
