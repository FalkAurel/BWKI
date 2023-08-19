import numpy as np

#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
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
        if depth <= self.maxDepth and nSamples >= self.minSampleSplit:#prePruning
            bestSplit = self.searchBestSplit(data, nFeatures)
            if bestSplit["informationGain"] > 0:
                leftSubTree = self.createTree(bestSplit["leftDataset"], depth= depth + 1)
                rightSubTree = self.createTree(bestSplit["rightDataset"], depth = depth + 1)
                return Node(condition=bestSplit["condition"], index=bestSplit["index"], informationGain=bestSplit["informationGain"],
                            left=leftSubTree, right=rightSubTree)
        leaf = self.majorityVoting(data[:, -1])
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
    
    def majorityVoting(self, y):
        y = list(y)
        return max(y, key=y.count)
    
    def predict(self, sample, Node, postPruning=False):
        """
        Traverses the tree in a similar fashion, as you'd traverse a binary Tree.
        """
        if postPruning != False:
            postPruning.append(Node)
        if Node.data != None:
            return Node.data
        check = sample[Node.index]
        if check < Node.condition:
            if postPruning:
                return self.predict(sample, Node.left, postPruning)
            return self.predict(sample, Node.left)
        else:
            if postPruning:
                return self.predict(sample, Node.right, postPruning)
            return self.predict(sample, Node.right)
        
    def evaluate(self, X, y,* , Node=None, predictions=False):
        """
        Evaluating the Tree. By default it returns the percentage to which the model is able to predict correctly. Alter-
        natively you can set predictions to True to get the predictions.
        """
        if Node == None:
            yhat = [self.predict(x, self.root) for x in X]
        else:
            yhat = [self.predict(x, Node) for x in X]
        if predictions:
            return yhat
        return np.mean(yhat == y)
    
    def postPruning(self, X, y):
        """
        creating a stack to track the Nodes the sample takes to get to the solution. We reverse the stack so that the
        first element becomes the leaf node, which makes it much easier to recursively traverse the tree. When we find a
        unique tree that is pruneable, we do it. We store the pruned tree in possiblePrunedTrees, before checking each
        element in possiblePrunedTrees to find the best of all of them.
        """
        import copy
        possiblePrunedTrees = []
        unPrunedPerformance = self.evaluate(X, y)
        for index, x in enumerate(X):
            stack = []
            yhat = self.predict(x, self.root, postPruning=stack)
            if yhat == y[index]:
                temporaryStack = copy.deepcopy(list(reversed(stack)))
                result = self.evaluatingStack(X, y, temporaryStack, unPrunedPerformance)
                if result:
                    possiblePrunedTrees.append(result)
        self.root = self.searchBestTree(X, y, possiblePrunedTrees)
        return self
        
    def depthFirstSearch(self, Node, stack = [], data = []):
        if Node.data:
            data.append(Node.data)
        if Node.left:
            stack.append(Node.left)
        if Node.right:
            stack.append(Node.right)
        if stack:
            return self.depthFirstSearch(stack.pop(), stack, data)
        return data
                
    def evaluatingStack(self, X, y, copyOfStack, baseLinePerformance):
        index = 1
        performance = float("inf")
        while index < len(copyOfStack) and performance >= baseLinePerformance:
            result = self.depthFirstSearch(copyOfStack[index])
            #print(result)
            copyOfStack[index].data = self.majorityVoting(result)
            copyOfStack[index].left, copyOfStack[index].right = None, None
            accuracy = self.evaluate(X, y, Node=copyOfStack[-1])
            #print(performance, accuracy)
            performance = accuracy
            if performance >= baseLinePerformance:
                print(True)
            #print(f"Updated performance {performance}")
            index += 1
        return copyOfStack
            
    def searchBestTree(self, X, y, possibleTrees):
        """
        Applying a greedy search algorithm to go through all the possibleTrees and select the best one which is then returned
        (at least it's rootNode).
        """
        maxAccuracy = -float("inf")
        bestTree = None
        for tree in possibleTrees:
            currentAccuracy = self.evaluate(X, y, Node=tree[-1])
            if currentAccuracy > maxAccuracy:
                maxAccuracy = currentAccuracy
                print(maxAccuracy)
                bestTree = tree[-1]
        return bestTree

if __name__ == "__main__":
    np.random.seed(100)
    X = np.random.randn(100, 9)
    y = np.zeros(100)
    y[(X[:, 0] + X[:, 1] > 1)] = 1
    y[(X[:, 0] + X[:, 1] < -1)] = 2
    index = int(len(X) * (2/3))
    trainX, trainY = X[:index], y[:index]
    t = Tree(minSampleSplit=3, maxDepth=X.shape[-1]*4, criterion="gini")
    print("Accuracy on training data without post pruning:", t.fit(trainX, trainY).evaluate(trainX, trainY))
    print("Accuracy on training data with post pruning:", t.postPruning(trainX, trainY).evaluate(trainX,trainY))
