from MovieLens import MovieLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator
import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Carregando os filmes")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputando avaliações")
    rankings = ml.getPopularityRanks()
    return(ml, data, rankings)

np.random.seed(42)
random.seed(42)

#Load up commom data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

#Construindo um objeto Evaluator para avaliar os algoritmos
evaluator = Evaluator(evaluationData, rankings)

#SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

#SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

#fazendo predicoes aleatorias 
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(True)