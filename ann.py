import tensorflow as tf
from tensorflow.examples.tutorials.mnist  import input_data

#Loading the dataset
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100
n_nodes_hl4 = 100

n_classes = 10
batch_size = 100


# PLACEHOLDERS
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,[None,10])

def neural_network(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_layer_4 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                      'bias': tf.Variable(tf.random_normal([n_classes]))}

    #y=xW+b

    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['bias'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['bias'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['bias'])
    l3 = tf.nn.sigmoid(l3)

    l4 = tf.add(tf.matmul(l3, hidden_layer_4['weights']), hidden_layer_4['bias'])
    l4 = tf.nn.sigmoid(l4)

    output = tf.matmul(l4,output_layer['weights']) + output_layer['bias']
    return output
"""
#VARIABLES
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#GRAPH OPERATIONS
y = tf.matmul(x,W)+b

#LOSS FUNCTION

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits=y))


#OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=0.5)

train = optimizer.minimize(cross_entropy)
"""
def train_neural_network(x):
    y_pred = neural_network(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)

    #CREATE SESSION
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for step in range(4000):
            batch_x,batch_y = mnist.train.next_batch(100)
            sess.run(train,feed_dict={x:batch_x,y_true:batch_y})

            if(step%100==0):
                #EVALUATE MODEL
                pred = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
                acc = tf.reduce_mean(tf.cast(pred,tf.float32))
                print(" Accuracy  After ",step,"  Epoch " )
                print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
                print('\n')

train_neural_network(x)