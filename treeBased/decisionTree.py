import numpy as np

class Node():
    def __init__(self, condition = None, index = None, informationGain =None, left = None, right = None, data = None):
        self.condition = condition
        self.index = index
        self.informationGain = informationGain
        self.left, self.right = left, right
        self.data = data
        
class Tree():
    def __init__(self, minSampleSplit, maxDepth, *, criterion="entropy"):
        self.minSampleSplit, self.maxDepth, self.criterion = minSampleSplit, maxDepth, criterion
        
    def fit(self, X, y):
        data = np.hstack((X, y.reshape(len(X), -1)))
        self.root = self.createTree(data)
        return self
    
    def createTree(self, data, *, depth=0):
        """
        recursively building a tree. Pretty similar to a binary Tree.
        """
        nFeatures = data.shape[-1]
        nSamples = len(data)
        if depth <= self.maxDepth and nSamples >= self.minSampleSplit:
            bestSplit = self.searchBestSplit(data, nFeatures)
            if bestSplit["informationGain"] > 0:
                leftSubTree = self.createTree(bestSplit["leftDataset"], depth= depth + 1)
                rightSubTree = self.createTree(bestSplit["rightDataset"], depth = depth + 1)
                return Node(condition=bestSplit["condition"], index=bestSplit["index"], informationGain=bestSplit["informationGain"],
                            left=leftSubTree, right=rightSubTree)
        leaf = self.evaluateY(data[:, -1])
        return Node(data=leaf)     
    
    def searchBestSplit(self, data, nfeatures):
        """
        searching for the best split by applying a greedy search algorithm.
        """
        currentMaxInformationGain, bestSplit = -float("inf"), {}
        for possibleConditionIndex in range(nfeatures - 1):#the last feature is the label so that wouldn't work
            uniqueConditions = np.unique(data[:, possibleConditionIndex])
            for condition in uniqueConditions:
                leftData, rightData = self.split(data, possibleConditionIndex, condition)
                if len(leftData) > 0 and len(rightData) > 0:
                    currentInformationGain = self.informationGain(data[:, -1], leftData[:,-1], rightData[:, -1], self.criterion)
                    if currentInformationGain > currentMaxInformationGain:
                        bestSplit["informationGain"] = currentInformationGain
                        bestSplit["index"] = possibleConditionIndex
                        bestSplit["condition"] = condition
                        bestSplit["leftDataset"] = leftData
                        bestSplit["rightDataset"] = rightData
        return bestSplit
    
    def split(self, data, index, condition):
        right = []
        left = []
        for sample in data:
            if sample[index] < condition:
                left.append(sample)
            else:
                right.append(sample)
        return np.array(left), np.array(right)
    
    def entropy(self, y):
        """
        Computes the Entropy. Entropy is expressed as E = Σ-p𝒊log(p𝒊), where p denotes the probability of class 𝒊
        """
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            probability = np.mean(np.where(y == label, 1, 0))
            entropy += -probability * np.log2(probability)
        return entropy
    
    def gini(self, y):
        """
        Computes the relative concentration of the labels more efficiently than Entropy. The Gini Index is expressed as:
        Gini = 1 - sum(p𝒊^2) for 𝒊 = 1 to n, where p denotes the probability of class 𝒊
        """
        labels = np.unique(y)
        gini = 0
        for label in labels:
            gini += (np.mean(np.where(label == y, 1, 0)))**2
        return 1 - gini

    def informationGain(self, parent, leftChild, rightChild, mode):
        """
        Computes the information gain. The bigger, the better. Information gain is computed with IG: E(parent) - Σω𝒊E(child𝒊)
        You can switch between gini and entropy by adjusting the mode parameter.
        """
        parentSize = len(parent)
        if mode == "gini":
            giniParent = self.gini(parent)
            giniLeftChild = self.gini(leftChild)
            giniRightChild = self.gini(rightChild)
            return giniParent - (len(leftChild) / parentSize * giniLeftChild + len(rightChild) / parentSize * giniRightChild)
        entropyParent = self.entropy(parent)
        entropyLeftChild = self.entropy(leftChild)
        entropyRightChild = self.entropy(rightChild)
        return entropyParent - (len(leftChild) / parentSize * entropyLeftChild + len(rightChild) / parentSize * entropyRightChild)
    
    def evaluateY(self, y):
        y = list(y)
        return max(y, key=y.count)
    
    def predict(self, sample, Node):
        """
        Traverses the tree in a similar fashion, as you'd traverse a binary Tree.
        """
        if Node.data != None:
            return Node.data
        check = sample[Node.index]
        if check < Node.condition:
            return self.predict(sample, Node.left)
        else:
            return self.predict(sample, Node.right)
        
    def evaluate(self, X, y,* , predictions=False):
        """
        Evaluating the Tree. By default it returns the percentage to which the model is able to predict correctly. Alter-
        natively you can set predictions to True to get the predictions.
        """
        yhat = [self.predict(x, self.root) for x in X]
        if predictions:
            return yhat
        return np.mean(yhat == y)

if __name__ == "__main__":
    X = np.random.randn(50, 9)
    y = np.zeros(50)
    y[(X[:, 0] + X[:, 1] > 1)] = 1
    y[(X[:, 0] + X[:, 1] < -1)] = 2
    tree = Tree(minSampleSplit=3, maxDepth=X.shape[-1]*4, criterion="entropy")
    print("Accuracy on training data:", tree.fit(X, y).evaluate(X, y))