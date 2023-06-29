import numpy as np
import matplotlib.pyplot as plt

def sparseToOneHotEncoded(y, nlabel):
    ausgabe = []
    for i in y:
        target = np.zeros(nlabel,)
        target[i] = 1
        ausgabe.append(target)
    return np.array(ausgabe)

def visualize(accuracy, loss, learningRate):
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
    ax3.set_title("Accuracy")
    plt.tight_layout()
    plt.show()