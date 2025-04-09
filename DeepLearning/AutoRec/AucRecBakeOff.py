from Evaluator import Evaluator
from MovieLens import MovieLens
from AutoRecAlgorithm import AutoRecAlgorithm
import numpy as np
import random
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def LoaMoviesData():
    ml = MovieLens()
    print('Carregando as avaliações dos filmes')
    data = ml.loadMovieLensLatestSmall()
    print('Coputando avaliações dos filmes')
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(42)
random.seed(42)

#Carregando os dados para o algoritmo de recomendaçãp
(ml, evaluationData, rankings) = LoaMoviesData()

#Construtor do algoritmo de evolução
evaluator = Evaluator(evaluationData, rankings)

#Autoencoder
AutoRec = AutoRecAlgorithm()
evaluator.AddAlgorithm(AutoRec, 'AutoRec')