"""

 The code was adapted from the original denoising auto-encoders (dA) tutorial
 in Theano.

 Contact Minmin Chen at chenmm24@gmail.com  if you have any questions.

 References :
   - M. Chen, K. Weinberger, F. Sha, Y. Bengio: Marginalized Denoising Auto-encoders
   for Nonlinear Representations, ICML 2014.

"""

import cPickle
import gzip
import os
import sys
import time
import scipy.io

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images, load_data

from PIL import Image

input_ll=0

root_folder = '../'

class mdA(object):
    """marginalized Denoising Auto-Encoder class (mdA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=1000,
                 W=None, bhid=None, bvis=None):
        """
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone mdA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the mdA and another architecture; if mdA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong mdA and another
                     architecture; if mdA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, noisemodel, noiserate, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the mdA """

        y = self.get_hidden_values(self.x)
	z = T.clip(self.get_reconstructed_input(y), 0.00247262315663, 0.997527376843)

	L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        dy = y * (1 - y)
        dz = z * (1 - z)

	df_x_2 = T.dot(T.dot(dz, self.W * self.W) * dy * dy,  self.W_prime * self.W_prime)

	if noisemodel == 'dropout':
		x_2 = self.x * self.x
		L2 = noiserate / (1 - noiserate) * T.mean(T.sum(df_x_2 * x_2, axis=1))
	else:
		L2 = noiserate * noiserate * T.mean(T.sum(df_x_2, axis=1))

        cost = T.mean(L) + 0.5 * L2

        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)


def test_mdA(data_name, learning_rate, noisemodel, noiserange, training_epochs=50, batch_size=20):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    tunning_epochs = range(1,20,2)+range(20, 301, 20);
    dataset = os.path.join(root_folder+'/data/layer'+str(input_ll), data_name + '.pkl.gz')
    print dataset
    output_folder = os.path.join(root_folder, 'params/mda_'+noisemodel+'/layer'+str(input_ll+1), data_name)
    if not os.path.isdir(output_folder):
    	os.makedirs(output_folder)

    filter_folder = os.path.join(root_folder, 'filters/mda_'+noisemodel+'/layer'+str(input_ll+1), data_name)
    if not os.path.isdir(filter_folder):
        os.makedirs(filter_folder)

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    print train_set_x.get_value(borrow=True).shape[0], train_set_x.get_value(borrow=True).shape[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    if not os.path.isdir(filter_folder):
        os.makedirs(filter_folder)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for rate in noiserange:

    	rng = numpy.random.RandomState(123)
    	theano_rng = RandomStreams(rng.randint(2 ** 30))

    	mda = mdA(numpy_rng=rng, theano_rng=theano_rng, input=x,
        	    n_visible=28*28, n_hidden=1000)

    	cost, updates = mda.get_cost_updates(noisemodel=noisemodel, noiserate=rate,learning_rate=learning_rate)

    	train_mda = theano.function([index], cost, updates=updates,
        	 givens={x: train_set_x[index * batch_size:
                	                (index + 1) * batch_size]})

    	start_time = time.clock()

    	############
    	# TRAINING #
    	############

    	# go through training epochs
    	for epoch in range(1, training_epochs+1):
        	# go through trainng set
        	c = []
        	for batch_index in xrange(n_train_batches):
			l = train_mda(batch_index)
            		c.append(l)

        	print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

		if epoch in tunning_epochs:
    			end_time = time.clock()
    			training_time = (end_time - start_time)
			print 'running time = %.2fm' % ((training_time) / 60.)

			output_filename = 'lrate='+str(learning_rate)+',noise='+str(rate)+',epoch='+str(epoch)+'.mat'
			scipy.io.savemat(os.path.join(output_folder, output_filename), {'W': mda.W.get_value(borrow=True), 'b':mda.b.get_value(borrow=True), 'b_prime':mda.b_prime.get_value(borrow=True)})

			filter_filename = 'lrate='+str(learning_rate)+',noise='+str(rate)+',epoch='+str(epoch)+'.png'
                        w_part = mda.W.get_value(borrow=True).T
                        w_part = w_part[0:900, :]
                        image = Image.fromarray(
                                        tile_raster_images(w_part,
                                        img_shape=(28, 28), tile_shape=(30, 30),
                                        tile_spacing=(1, 1)))
                        image.save(os.path.join(filter_folder, filter_filename))

	print 'The corruption code for file '+os.path.split(__file__)[1]+' with noise level %.2f' % (rate) + ' learning rate %.6f' %(learning_rate)  + ' finished in %.2fm' % ((training_time) / 60.)



if __name__ == '__main__':

    if len(sys.argv) != 4:
        print 'Usage: python pretrain_mda.py noiseModel dataset learningRate'
        print 'Example: python pretrain_mda.py gauss basic 0.1'
        sys.exit()

    mm = sys.argv[1]
    dd = sys.argv[2]
    lr = float(sys.argv[3])
    log_folder = root_folder+'/logs/mda_'+mm+'/layer'+str(input_ll+1)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)
    logfile = open(os.path.join(log_folder, dd+',lr='+str(lr)+'.log'), 'a', 0)
    sys.stdout = logfile
    if mm == 'dropout':
        noiserange = [0.1, 0.25, 0.40, 0.55, 0.7, 0.85]
    else:
        noiserange = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    test_mdA(data_name=dd, learning_rate=lr, noisemodel=mm, noiserange = noiserange)
    sys.stdout = sys.__stdout__
