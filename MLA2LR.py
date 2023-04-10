import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer


# This is our logistic function. It is called in both our predict and gradient_descent methods as an activation function
def logistic_func(x1):
    return 1 / (1 + np.exp(-x1))


# This is our gradient descent method. It works by first getting the shape of our x training set. It then runs a for
# loop for the number of epochs specified. Next it takes the dot product of our training set and the weights. This
# is then plugged into our logistic function. Next, we calculate our error and use this error to calculate our gradient
# Lastly, we update our weights and return the final form of them.
def gradient_descent(train_x, train_y, weight, lr, epochs):
    m, n = train_x.shape
    for i in range(epochs):
        predict1 = np.dot(train_x, weight)
        predicted = logistic_func(predict1)
        error = predicted - train_y
        grad_w = np.dot(train_x.T, error) / m
        weight = weight - lr * grad_w
    return weight


# This is our predict method. It is responsible for making predictions on our testing set. It works by first taking the
# dot product of our x training set and our updated weights. It then plugs this number into our logistic function.
# Lastly, it replaces every prediction greater than or equal to 0.5 with a 1 and anything less with a 0 before returning
# our predictions.
def predict(x_set, w):
    pred_1 = np.dot(x_set, w)
    pred_2 = logistic_func(pred_1)
    predictions = np.where(pred_2 >= 0.5, 1, 0)
    return predictions


# This is our class accuracy method. It is responsible for determining our classification accuracy and returning it.
# It works by getting the length of our predicted set and adding a 1 to a counter variable everytime it finds a correct
# prediction. Lastly, our counter variable divided by our length.
def class_accuracy(actual, predicted):
    m = len(predicted)
    count = 0
    for p in range(m):
        if predicted[p] == actual[p]:
            count = count + 1
    return count / m


# This is our main method. It works by first loading in our dataset and then getting our data set prepared to be
# worked on by doing things such as adding a row of 1s (for our intercept) and normalizing our x sets. Next it calls
# gradient descent to update our weights, then predict to predict our testing set and lastly, class accuracy to print
# our accuracy
data = load_breast_cancer()
list(data.target_names)
['malignant', 'benign']
x = np.array(data.data)
y = np.array(data.target)
x_rows, x_columns = x.shape
intercepts = np.ones(x_rows)
intercepts = intercepts.reshape((569, 1))
updated_x = np.hstack((x, intercepts))
x_train = np.array(updated_x[0:398])
y_train = np.array(y[0:398])
x_test = np.array(updated_x[398:])
y_test = np.array(y[398:])
rows, columns = x_train.shape
weights = np.array(np.zeros(columns))
weights = weights.reshape((31, 1))
y_train = y_train.reshape((398, 1))
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
weights = gradient_descent(x_train, y_train, weights, 0.05, 100000)
f_predictions = predict(x_test, weights)
score = class_accuracy(y_test, f_predictions)
print()
print("Classification Accuracy: ")
print(score)
