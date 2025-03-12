import numpy as np
import tensorflow as tf

class AutoRec:
    def __init__(self, visibleDimmensiosn = 50, epochs = 200, hiddenDimensions = 50, learning_rate = 0.1, batchSize = 100):
        self.visibleDimensions = visibleDimmensiosn
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def Train(self, X):
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batchSize):
                x_batch = X[i:i+self.batchSize]
                self.train_step(x_batch)
            print("Epoch: ", epoch, " Loss: ", self.loss(x_batch))
    
    def GetRecommendations(self, inputUser): #Innput user gerado para erar as previsoes para um usuario
        #Feed through a single user and return predictions fro the output layer
        rec = self.neural_net(inputUser)
        #Usado para retornar as recomendações
        return rec[0]
    
    def neural_net(self, inputUser):
        tf.random.set_seed(42)
        #Create variables for weights and biases
        self.weights = {
            'h1': tf.Variable(tf.random.normal([self.visibleDimensions, self.hiddenDimensions])),
            'out': tf.Variable(tf.random.normal([self.hiddenDimensions, self.visibleDimensions]))
        }

        #Create biases
        self.biases = {
            'b1': tf.Variable(tf.random.normal([self.hiddenDimensions])),
            'out': tf.Variable(tf.random.normal([self.visibleDimensions]))
        }

        #Criando a camada de entrada
        self.inputLayer = inputUser
        #Criando a camada oculta
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.inputLayer, self.weights['h1']), self.biases['b1']))# Multiplica a camada de entrada com os pesos e adiciona o bias
        #Output layer
        self.outputLayer = tf.nn.sigmoid(tf.add(tf.matmul(hidden, self.weights['out']), self.biases['out']))

        return self.outputLayer
    
    def run_optimization(self, inputUser):
        with tf.GradientTape() as g:
            pred = self.neural_net(inputUser)
            loss = self.loss(pred, inputUser)

        trainable_variables = list(self.weights.values()) + list(self.biases.values())
        gradients = g.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
