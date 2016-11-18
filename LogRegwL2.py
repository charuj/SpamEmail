% Logistic regression with L2 Regularization 


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def xavier(n_inputs,n_outputs):
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform([n_inputs,n_outputs],-init_range, init_range)


# Loading dataset
import csv
with open('spambase.train.txt', 'rb') as csvfile:
 spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
 count= 0
 datarrr = []
 for row in spamreader:
     if count == 0 or count==1:
        count += 1

     else:
         count +=1

         datarrr.append(row)
datarrr=np.array(datarrr)

for row in datarrr:
    try:
        row.astype(np.float32)
    except:
        print row
datarrr=datarrr.astype(np.float32)

targets= datarrr[:, -1]
targets=targets.reshape([3000, 1])
inputs= datarrr[:, :-1]

# binarize training data

for i in range(inputs.shape[0]):
     for j in range(inputs.shape[1]):
         if inputs[i,j] != 0:
             inputs[i,j]=1



N=inputs.shape[0]

'''inputs= np.log(inputs + 1)'''


'''
# Normalizing training data to have zero mean, variance 1
inputs= np.asarray(inputs)
input_mean= inputs.mean(axis=0)
input_std= inputs.std(0)
inputs= (inputs- input_mean)/input_std# this is supposed to normalize the inputs
'''

# importing test data

# Loading dataset
import csv
with open('spambase.test.txt', 'rb') as csvfile:
 spamreader_test = csv.reader(csvfile, delimiter=',', quotechar='|')
 count= 0
 test_data = []
 for row in spamreader_test:
     if count == 0 or count==1:
        count += 1

     else:
         count +=1

         test_data.append(row)
test_data=np.array(test_data)

for row in test_data:
    try:
        row.astype(np.float32)
    except:
        print row
test_data=test_data.astype(np.float32)

test_targets= test_data[:, -1]
m= test_data.shape[0]
test_targets=test_targets.reshape([m, 1])
test_inputs= test_data[:, :-1]

# binarize training data

for i in range(test_inputs.shape[0]):
     for j in range(test_inputs.shape[1]):
         if test_inputs[i,j] != 0:
             test_inputs[i,j]=1


'''test_inputs=np.log(test_inputs + 1)'''

'''
# Normalizing test data to have zero mean, variance 1, using the mean and variance of the training data
test_inputs= np.asarray(test_inputs)
test_inputs= (test_inputs- input_mean)/input_std# this is supposed to normalize the inputs
'''

# Creating placeholders and variables

# Placeholders
X= tf.placeholder("float", shape=[None, 57]) # confirm its 57
Y= tf.placeholder("float", shape=[None,1]) # confirm that this is referring to labels
#R= tf.placeholder("float", shape=[1])# shape of regularizer is one because weight multiplication gives a scaler

#Variables
W= tf.Variable(xavier(57,1))
b= tf.Variable(tf.ones(shape=[1]))

X.get_shape()

# creating the logistic regression model  sigma(XtW + b)
logits=tf.add(tf.matmul(X,W),b)
output= tf.nn.sigmoid(logits) # putting the logits through the sigmoid activation function

# cross entropy loss function

ce= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost= tf.reduce_mean(ce) # what is the point of this, is it to reduce the CE? Doesn't an optimizer reduce CE

# l2 regularization
regularizers= tf.nn.l2_loss(W) +tf.nn.l2_loss(b)

# adding regularizer to cost
loss= tf.add(cost, regularizers*.0)

# optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
#train_optimization= tf.train.Optimizer.minimize(loss) # minimizing cross entropy with the added regularizer

# compute accuracy

''' creating threshhold, if the predicted output is greater than 0.5 (i.e. is 1)
then the statement returns true (1) else when the output is less than 0.5 (i.e. 0)
it returns false (0). Notice that the boolean has the same value as the predicted output.
'''

prediction= tf.greater(output, 0.5)
prediction_float= tf.cast(prediction, "float")


# computing accuracy

correct=0
result= tf.equal(prediction_float, Y)# if it is true that the prediction equals the target, then add 1
result2 = tf.cast(result, "float")
correct= tf.reduce_mean(result2)



# error rate
#error_rate = tf.to_float(1.0 - tf.to_float(correct/tf.shape(X))[0]) # compare this to ECE521
error_rate = correct

# launch the session
sess= tf.InteractiveSession()

# cross validation here (use old code), average the error rate for the 4 training sets

# 5 fold cross-validation

# Divide training set into 5 chunks

set1= inputs[0:N/5]
set2= inputs[N/5: 2*N/5]
set3= inputs[2*N/5: 3*N/5]
set4= inputs[3*N/5: 4*N/5]
set5= inputs[4*N/5:]

# Divide the target set into 5 chunks

set1target= targets[0:N/5]
set2target=targets[N/5: 2*N/5]
set3target= targets[2*N/5: 3*N/5]
set4target= targets[3*N/5: 4*N/5]
set5target= targets[4*N/5:]

# Loop for validation set

validation_error_sum= 0
init= tf.initialize_all_variables()

for i in range(0,5):
    input_set = [set1,set2,set3,set4,set5]
    target_set= [set1target, set2target, set3target, set4target, set5target]
    valid= input_set.pop(i)
    training_set = np.concatenate(input_set,axis=0)
    valid_target = target_set.pop(i)
    training_target= np.concatenate(target_set, axis=0)


    # initialize the variables

    sess.run(init)
    accuracy= sess.run(correct,feed_dict={X:inputs, Y:targets})
    print "Initialized"
    print "Error Rate " + str( 1- accuracy)



    for epoch in range(150):
        sess.run(optimizer, feed_dict={X:training_set, Y:training_target})
        accuracy= sess.run(correct,feed_dict={X:training_set, Y:training_target})
        accuracy_valid= sess.run(correct, feed_dict={X: valid, Y:valid_target})

        training_error_rate= 1- accuracy

        validation_error_rate= 1- accuracy_valid

    validation_error_sum += validation_error_rate



    #print "Training Accuracy " + str(accuracy)
    print "Training Error Rate" + str(training_error_rate)
    print "Validation Error Rate" + str(validation_error_rate)


weights= sess.run(W)
print str(weights)
weights= np.argsort(weights.flatten())
print str(weights)



avg_valid_error= validation_error_sum/5
print "Average Validation Error for Session " + str(avg_valid_error)

test_correct = sess.run(correct,feed_dict={X:test_inputs, Y:test_targets})
test_error= 1.0- test_correct
print "Test Error with Regularizer of 0.01 = "   + str(test_error)
