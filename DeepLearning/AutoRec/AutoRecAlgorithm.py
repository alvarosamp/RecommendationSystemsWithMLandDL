from surprise import AlgoBase # Base class for all algorithms
from surprise import PredictionImpossible
import numpy as np
from AutoRec import AutoRec

class AutoRecAlgorithm(AlgoBase):
    def __init__(self, epochs = 100, hiddenDim = 100, learning_rate = 0.01, batchSize = 100, sim_options = {}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        numUsers = trainset.n_users
        numItems = trainset.n_items

        trainingMatrix = np.zeros([numUsers, numItems], dtype = np.float32)  # Cria uma matriz para a avaliação que inicialmente é zero

        for (uid, iid, rating) in trainset.all_ratings():
            trainingMatrix[int(uid), int(iid)] = rating / 5.0

        # Cria uma instância de AutoRec com os parâmetros fornecidos
        autoRec = AutoRec(trainingMatrix.shape[1], self.epochs, self.hiddenDim, self.learning_rate, self.batchSize)
        autoRec.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype = np.float32)

        for uiid in range(trainset.n_users):
            if uiid % 50 == 0:
                print("Processing user ", uiid)
            recs = autoRec.GetRecommendations([trainingMatrix[uiid]])

            for itemID, rec in enumerate(recs):
                self.predictedRatings[uiid, itemID] = rec * 5.0

        return self