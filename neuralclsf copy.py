import tensorflow as tf
import keras
#import cv2
import os
import imageio
import sys
import numpy as np
def reshape(matrix):
    flaggy=[]
    for i in range(0,127):
        for j in range(0,127):
            for k in range(0,3):
                troo=matrix[i,j,k]
                troo2=4.56454
                troo2=troo/255.0
                flaggy.append(troo2)
    flame=np.asarray(flaggy)
    return flame
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
classific1=load_images_from_folder("classif1")
classific2=load_images_from_folder("classif2")

classificationst=classific1+classific2
features=[]
for i in range(0,len(classificationst)):
    t=np.asarray(classificationst[i])
    features.append(t)
input_images=np.asarray(features)
input_images=input_images/255.0
labels=[]
for io in range (0,len(classific1)):
    labels.append(0)
for ip in range (0,len(classific2)):
    labels.append(1)
target_labels=np.asarray(labels)
#input_images = tf.placeholder(tf.float32, shape=[None, 48387])
#target_labels = tf.placeholder(tf.float32, shape=[None, 2])
#hidden_nodes = 512

#input_weights = tf.Variable(tf.truncated_normal([48387, hidden_nodes]))
#input_biases = tf.Variable(tf.zeros([hidden_nodes]))

#hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 2]))
#hidden_biases = tf.Variable(tf.zeros([2])
#input_layer = tf.matmul(input_images, input_weights)

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(127,127,3)),
                          keras.layers.Dense(512, activation=tf.nn.relu),
                          keras.layers.Dense(256, activation=tf.nn.relu),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(64, activation=tf.nn.relu),
                          keras.layers.Dense(16, activation=tf.nn.relu),
                          keras.layers.Dense(2, activation=tf.nn.softmax),
                          ])



model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(input_images,target_labels,epochs=20)



classific10=load_images_from_folder("test1")
classific20=load_images_from_folder("test2")

classificationstza=classific10+classific20
features2=[]
for i in range(0,len(classificationstza)):
    t=classificationstza[i]
    features2.append(t)
test_images1748=np.asarray(features2)
test_images1748=test_images1748/255.0



test_labels1=[]
for io in range (0,len(classific10)):
    test_labels1.append(0)
for ip in range (0,len(classific20)):
    test_labels1.append(1)
test_labels=np.asarray(test_labels1)

#hidden_layer = tf.nn.relu(input_layer + input_biases)
#digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases
                            
#loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))
                            
#optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)
                            
#correct_prediction = tf.equal(tf.argmax(digit_weights,1), tf.argmax(target_labels,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_loss, test_acc = model.evaluate(test_images1748, test_labels)

print('Test accuracy:', test_acc)
model.save('arth.h5')
