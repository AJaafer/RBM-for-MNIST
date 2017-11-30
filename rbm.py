# credits to: https://github.com/patricieni/RBM-Tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def sample(probabilities):
    ''' Sample a tensor based on the probabilities (A tensor given by get_probabilities)'''
    return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), 0, 1))


# Simple RBM class
# Designed to be modular, should attach the weights as well, will be helpful when dealing with DBM
class RBM:
    def __init__(self, n_visible, n_hidden, lr, epochs, batch_size=None):
        ''' Initialize a model for an RBM with one layer of hidden units '''
        self.n_hidden = n_hidden #  Number of hidden nodes
        self.n_visible = n_visible # Number of visible nodes
        self.lr = lr # Learning rate for the CD algorithm
        self.epochs = epochs # Number of iterations to run the algorithm for
        self.batch_size = batch_size # Split the training data

        # Initialize weights and biases
        with tf.name_scope('Weights'):
            self.W = tf.Variable(tf.random_normal([self.n_visible, self.n_hidden], mean=0., stddev=4 * np.sqrt(6. / (self.n_visible + self.n_hidden))), name="weights")
        tf.summary.histogram('weights',self.W)
        self.vb = tf.Variable(tf.zeros([1, self.n_visible]),tf.float32, name="visible_bias")
        self.hb = tf.Variable(tf.zeros([1, self.n_hidden]),tf.float32, name="hidden_bias")


    def get_probabilities(self, layer, val):
        ''' Return a tensor of probabilities associated with the layer specified'''
        # val: Input units, hidden or visible as binary or float
        if layer == 'hidden':
            with tf.name_scope("Hidden_Probabilities"):
                return tf.nn.sigmoid(tf.matmul(val, self.W) + self.hb)
        elif layer == 'visible':
            with tf.name_scope("Visible_Probabilities"):
                return tf.nn.sigmoid(tf.matmul(val, tf.transpose(self.W)) + self.vb)


    def gibbs(self, steps, v):
        ''' Use the Gibbs sampler for a network of hidden and visible units. Return a sampled version of the input'''
        with tf.name_scope("Gibbs_sampling"):
            for i in range(steps): # Number of steps to run the algorithm
                hidden_p = self.get_probabilities('hidden', v) # v: input data
                h = sample(hidden_p)

                visible_p = self.get_probabilities('visible', h)
                v = visible_p
                #v = sample(visible_p)
            return visible_p


    def plot_weight_update(self, x=None, y=None):
        plt.xlabel("Epochs")
        plt.ylabel("CD difference")
        plt.title('Weight increment change throughout learning')
        plt.plot(x, y, 'r--')
        plt.show()


    def free_energy(self, v):
        ''' Compute the free energy for a visible state'''
        vbias_term = tf.matmul(v, tf.transpose(self.vb))
        x_b = tf.matmul(v, self.W) + self.hb
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(x_b)))
        return - hidden_term - vbias_term