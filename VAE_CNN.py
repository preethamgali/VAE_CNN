import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
import os
from keras.datasets import mnist

def normalize_data(data):
  data[data<=100] = 0
  data[data>100] = 1
  return data 

def normalizing_layers_output(data):
  mean = tf.math.reduce_mean(data)
  std = tf.math.reduce_std(data)
  data = data - mean
  return data/std

def reshape_img(data):
    data = normalize_data(data)
    lis = []
    for i in data:
        lis.append(i.reshape((28,28,1)))
    return np.asarray(lis)

def reparametarzing(mu , log_sigma):
    epsilon = tf.random.truncated_normal(shape =tf.shape(mu),mean=0,stddev = 1)
    return mu + tf.math.exp(0.5 * log_sigma) * epsilon

def KL_classification_error():
  classification_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = input_img,logits = uconv0))
  KL_error = 0.5 * tf.reduce_sum(tf.exp(log_var) + tf.square(mu) - 1. - log_var)
  return classification_error+KL_error
  
latent_space = 5 
with tf.device('/GPU:0'):

#     [filter_height, filter_width, in_channels, out_channels]
  conv1_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,1,16],mean =0 ,stddev=1),name = "conv1_filters")
  conv1_bias = tf.Variable(tf.zeros(16),name="conv1_bias")

  conv2_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,16,8],mean =0 ,stddev=1),name = "conv2_filters")
  conv2_bias = tf.Variable(tf.zeros(8),name="conv2_bias")

  conv3_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,8,8],mean =0 ,stddev=1),name = "conv3_filters")
  conv3_bias = tf.Variable(tf.zeros(8),name="conv3_bias")

  uconv3_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,8,8],mean =0 ,stddev=1),name = "uconv3_filters")
  uconv3_bias = tf.Variable(tf.zeros(8),name="uconv3_bias")

  uconv2_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,8,8],mean =0 ,stddev=1),name = "uconv2_filters")
  uconv2_bias = tf.Variable(tf.zeros(8),name="uconv2_bias")

  uconv1_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,8,16],mean =0 ,stddev=1),name = "uconv1_filters")
  uconv1_bias = tf.Variable(tf.zeros(16),name="uconv1_bias")

  uconv0_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,16,1],mean =0 ,stddev=1),name = "uconv0_filters")
  uconv0_bias = tf.Variable(tf.zeros(1),name="uconv0_bias")

  fc1_mu_weights = tf.Variable(tf.random.truncated_normal(shape = [4*4*8,latent_space],mean =0 ,stddev=1))
  fc1_mu_weights_b = tf.Variable(tf.zeros(1))

  fc1_var_weights = tf.Variable(tf.random.truncated_normal(shape = [4*4*8,latent_space],mean =0 ,stddev=1))
  fc1_var_weights_b = tf.Variable(tf.zeros(1))
    
  d_fc1_weights = tf.Variable(tf.random.truncated_normal(shape = [latent_space,4*4*8],mean =0 ,stddev=1),name = "d_fc1_weights")
  d_fc1_weights_b = tf.Variable(tf.zeros(1))


  # 28x28 ---> 28x28x1
  input_img = tf.placeholder(tf.float32,(None,28,28,1),name = 'input')

  # 28x28x1 --> 28x28x16
  conv1 = tf.nn.convolution(input_img , conv1_filters, padding = "SAME", strides = [1,1,1,1],name="conv1")
  conv1 = tf.nn.bias_add(conv1,conv1_bias,name="conv1_b")
  conv1 = normalizing_layers_output(tf.nn.relu(conv1,name="conv1_relu"))
                            
# 28x28x16 --> 14x14x16
  conv1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding = 'SAME',name = "conv1_maxp")

  # 14x14x16 --> 14x14x8
  conv2 = tf.nn.convolution(conv1 , conv2_filters, padding = "SAME", strides = [1,1,1,1],name="conv2")
  conv2 = tf.nn.bias_add(conv2,conv2_bias,name="conv2_b")
  conv2 = normalizing_layers_output(tf.nn.relu(conv2,name="conv2_relu"))

# 14x14x8 ---> 7x7x8
  conv2 = tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding = 'SAME',name = "conv2_maxp")

  # 7x7x8 ---> 7x7x8
  conv3 = tf.nn.convolution(conv2 , conv3_filters, padding = "SAME", strides = [1,1,1,1],name="conv3")
  conv3 = tf.nn.bias_add(conv3,conv3_bias,name="conv3_b")
  conv3 = normalizing_layers_output(tf.nn.relu(conv3,name="encode"))
                            
# 7x7x8 ---> 4x4x8
  fc1_in = tf.nn.max_pool(conv3,[1,2,2,1],[1,2,2,1],padding = 'SAME',name = "encode")

# 4x4x8 ---> 1x128
  fc1_input = tf.reshape(fc1_in,shape = [-1,4*4*8])
    
# 1x128 ---> 1xlatent_space
  mu = normalizing_layers_output(tf.nn.relu(tf.matmul(fc1_input,fc1_mu_weights) + fc1_mu_weights_b))
  log_var = normalizing_layers_output(tf.nn.relu(tf.matmul(fc1_input,fc1_var_weights) + fc1_var_weights_b))



# 1xlatent_space
  latent_z = reparametarzing(mu , log_var)

# 1xlatent_space ---> 1x128
  decoder_fc1 = normalizing_layers_output(tf.nn.relu(tf.matmul(latent_z,d_fc1_weights) + d_fc1_weights_b))

# 1x128 ---> 4x4x8
  fc1_reshape = tf.reshape(decoder_fc1,shape = [-1,4,4,8])


# 4x4x8 --> 7x7x8
  uconv3 = tf.image.resize_nearest_neighbor(fc1_reshape,size=(7,7),name = "upscale1")
  uconv3 = tf.nn.convolution(uconv3 , uconv3_filters, padding = "SAME", strides = [1,1,1,1],name="uconv3")
  uconv3 = tf.nn.bias_add(uconv3,uconv3_bias,name="uconv3_b")
  uconv3 = normalizing_layers_output(tf.nn.relu(uconv3,name="uconv3_relu"))


# 7x7x8 --> 14x14x8
  uconv2 = tf.image.resize_nearest_neighbor(uconv3,size=(14,14),name = "upscale2")
  uconv2 = tf.nn.convolution(uconv2 , uconv3_filters, padding = "SAME", strides = [1,1,1,1],name="uconv2")
  uconv2 = tf.nn.bias_add(uconv3,uconv3_bias,name="uconv2_b")
  uconv2 = normalizing_layers_output(tf.nn.relu(uconv2,name="uconv2_relu"))
                            
# 14x14x8  ---> 28x28x16
  uconv1 = tf.image.resize_nearest_neighbor(uconv2,size=(28,28),name = "upscale3")
  uconv1 = tf.nn.convolution(uconv1 , uconv1_filters, padding = "SAME", strides = [1,1,1,1],name="uconv1")
  uconv1 = tf.nn.bias_add(uconv1,uconv1_bias,name="uconv1_b")
  uconv1 = normalizing_layers_output(tf.nn.relu(uconv1,name="uconv1_relu"))
                            
# 28x28x16  ---> 28x28x1
  uconv0 = tf.nn.convolution(uconv1 , uconv0_filters, padding = "SAME", strides = [1,1,1,1],name="uconv0")
  uconv0 = normalizing_layers_output(tf.nn.bias_add(uconv0,uconv0_bias,name="uconv1_b"))
  
  output = tf.nn.sigmoid(uconv0,name="sigmoid")


  error = KL_classification_error()
  t_error = tf.reduce_mean(error)
  train = tf.train.AdamOptimizer().minimize(t_error)
  
errors = []
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = reshape_img(X_train)
sess = tf.Session()
for _ in range(1):
  epoch = 10000
  sess.run(tf.global_variables_initializer())

  for e in range(epoch):
    count = 0
    out = sess.run(train,{input_img:X_train[:1000]})
    if e%1000 == 0:
      er = sess.run(t_error,{input_img:X_train[:100]})
      errors.append(er)
      print(e,"-epoch ",sum(errors)/len(errors),"Error for testing data")
      
z = np.random.normal(0,1,size=(1,latent_space))
print(z)
out = sess.run(output,{latent_z:z})
plt.imshow(out[0][:,:,0], cmap='gray')

sess.close()
