import csv
import random


#--------------------------------------------------------------------------------------
# Splitting the dataset into a training set and a test set

#opening the dataset file
with open ('diabetes_5050.csv', 'r', encoding="utf-8") as f:
  reader = csv.reader(f)
  rows = list(reader)

# Splitting the rows of the dataset depending on the outcome
rows0 = [i for i in rows if i[-1] == '0.0']
rows1 = [i for i in rows if i[-1] == '1.0']

# Creating the test dataset with the first 1000 rows of each outcome
test_data = rows0[:1000] + rows1[:1000]
rows0,rows1 = rows0[1000:] , rows1[1000:]

training_data = rows0 + rows1

# Writing the training set in a file 
with open('training_set.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    random.shuffle(training_data)
    writer.writerows(training_data)

# Writing the test set in a file
with open('test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    random.shuffle(test_data)
    writer.writerows(test_data)

#--------------------------------------------------------------------------------------

from skmultiflow.data import FileStream
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier, HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier  
import time

#--------------------------------------------------------------------------------------
# Extremely Fast Decision Tree

# Setting up the data streams
training_stream = FileStream('training_set.csv')
test_stream = FileStream('test.csv')

# Setup Extremely Fast Decision Tree classifier
efdt = ExtremelyFastDecisionTreeClassifier()


start = time.time()

# Setup variables to control loop and track performance
n_samples = 0
correct_cnt = 0
max_samples = 20000

# Train the estimator with the samples provided by the data stream
while n_samples < max_samples and training_stream.has_more_samples():
    X, y = training_stream.next_sample()
    efdt.partial_fit(X, y)
    n_samples += 1

n_test_samples = 0

# Test the estimator with the samples provided by the data stream
y_true = []
y_predicted = []

while n_test_samples < max_samples and test_stream.has_more_samples():
    X, y = test_stream.next_sample()
    y_pred = efdt.predict(X)
    y_true.append(y)
    y_predicted.append(y_pred)
    if y[0] == y_pred[0]:
        correct_cnt += 1
    n_test_samples += 1

end = time.time()
elapsed = end - start


# Display results

print ('Execution time : ', elapsed, 'ms')
print('{} samples analyzed.'.format(n_samples))

print(f"Tree depth : {efdt.get_model_measurements['Tree depth']}")
print(f"Number of nodes : {efdt.get_model_measurements['Tree size (nodes)']}")
print(f"Number of leaves : {efdt.get_model_measurements['Tree size (leaves)']}")

print('Extremely Fast Decision Tree accuracy: {}'.format(correct_cnt / n_test_samples))
target_names = ['non diabetes', 'diabetes']
print(classification_report(y_true, y_predicted, target_names=target_names))


#--------------------------------------------------------------------------------------
# Very Fast Decision Tree

# Setting up a data stream
training_stream = FileStream('training_set.csv')
test_stream = FileStream('test.csv')

# Setup Hoeffding Tree estimator
ht = HoeffdingTreeClassifier()

start = time.time()

# Setup variables to control loop and track performance
n_samples = 0
correct_cnt = 0
max_samples = 20000

# Train the estimator with the samples provided by the data stream
while n_samples < max_samples and training_stream.has_more_samples():
    X, y = training_stream.next_sample()
    ht.partial_fit(X, y)
    n_samples += 1

n_test_samples = 0

# Test the estimator with the samples provided by the data stream
y_true = []
y_predicted = []

while n_test_samples < max_samples and test_stream.has_more_samples(): 
  X, y = test_stream.next_sample()
  y_pred = ht.predict(X)
  y_true.append(y)
  y_predicted.append(y_pred)
  if y[0] == y_pred[0]:
      correct_cnt += 1
  n_test_samples += 1

end = time.time()
elapsed = end - start

# Display results

print ('Execution time : ', elapsed, 'ms')
print('{} samples analyzed.'.format(n_samples))

print(f"Tree depth : {ht.get_model_measurements['Tree depth']}")
print(f"Number of nodes : {ht.get_model_measurements['Tree size (nodes)']}")
print(f"Number of leaves : {ht.get_model_measurements['Tree size (leaves)']}")

print('Very Fast Tree accuracy: {}'.format(correct_cnt / n_test_samples))
target_names = ['non diabetes', 'diabetes']
print(classification_report(y_true, y_predicted, target_names=target_names))


#--------------------------------------------------------------------------------------
# Concept-adaptating Very Fast Decision Tree

# Setting up a data stream
training_stream = FileStream('training_set.csv')
test_stream = FileStream('test.csv')

# Setup Hoeffding Tree estimator
cvfdt = HoeffdingAdaptiveTreeClassifier()

start = time.time()

# Setup variables to control loop and track performance
n_samples = 0
correct_cnt = 0
max_samples = 20000

# Train the estimator with the samples provided by the data stream
while n_samples < max_samples and training_stream.has_more_samples():
    X, y = training_stream.next_sample()
    cvfdt.partial_fit(X, y)
    n_samples += 1

n_test_samples = 0

# Test the estimator with the samples provided by the data stream
y_true = []
y_predicted = []

while n_test_samples < max_samples and test_stream.has_more_samples(): 
  X, y = test_stream.next_sample()
  y_pred = cvfdt.predict(X)
  y_true.append(y)
  y_predicted.append(y_pred)
  if y[0] == y_pred[0]:
      correct_cnt += 1
  n_test_samples += 1

end = time.time()
elapsed = end - start

# Display results

print ('Execution time : ', elapsed, 'ms')
print('{} samples analyzed.'.format(n_samples))

print(f"Tree depth : {cvfdt.get_model_measurements['Tree depth']}")
print(f"Number of nodes : {cvfdt.get_model_measurements['Tree size (nodes)']}")
print(f"Number of leaves : {cvfdt.get_model_measurements['Tree size (leaves)']}")

print('Concept-adaptating Very Fast Tree accuracy: {}'.format(correct_cnt / n_test_samples))
target_names = ['non diabetes', 'diabetes']
print(classification_report(y_true, y_predicted, target_names=target_names))

#--------------------------------------------------------------------------------------
# Decision Tree

#training set
training_outcome = [i[-1] for i in training_data]
training_without_outcome = [i[:-1] for i in training_data]

#test set
test_expected_outcome = [i[-1] for i in test_data]
test_without_outcome = [i[:-1] for i in test_data]

# Créer un classificateur d'arbre de décision
dt = DecisionTreeClassifier()
start = time.time()
dt.fit(training_without_outcome, training_outcome)
test_predicted_outcome = dt.predict(test_without_outcome)

end = time.time()
elapsed = end - start

print ('Execution time : ', elapsed, 'ms')

print(f"Tree depth : {dt.get_depth()}")
print(f"Number of nodes : {dt.get_n_leaves()}")
print(f"Number of leaves : {dt.tree_.node_count}")

target_names = ['non diabetes', 'diabetes']
print(classification_report(test_expected_outcome, test_predicted_outcome, target_names=target_names))

