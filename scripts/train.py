import csv
import random

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier  
import time
from dvclive import Live

#--------------------------------------------------------------------------------------
# # Extremely Fast Decision Tree

# # Setting up the data streams
# training_stream = FileStream('training_set.csv')
# test_stream = FileStream('test.csv')

# # Setup Extremely Fast Decision Tree classifier
# efdt = ExtremelyFastDecisionTreeClassifier()


# start = time.time()

# # Setup variables to control loop and track performance
# n_samples = 0
# correct_cnt = 0
# max_samples = 200

# with Live() as live:

#     for i in range(10): 

#         # Train the estimator with the samples provided by the data stream
#         while n_samples < max_samples and training_stream.has_more_samples():
#             X, y = training_stream.next_sample()
#             efdt.partial_fit(X, y)
#             n_samples += 1

#         n_test_samples = 0

#         # Test the estimator with the samples provided by the data stream
#         y_true = []
#         y_predicted = []

#         while n_test_samples < max_samples and test_stream.has_more_samples():
#             X, y = test_stream.next_sample()
#             y_pred = efdt.predict(X)
#             y_true.append(y)
#             y_predicted.append(y_pred)
#             if y[0] == y_pred[0]:
#                 correct_cnt += 1
#             n_test_samples += 1

#         end = time.time()
#         elapsed = end - start

#         live.log_metric("accuracy",correct_cnt / n_test_samples)
#         live.log_metric("Tree depth",efdt.get_model_measurements['Tree depth'])
#         live.log_metric("Tree size (nodes)",efdt.get_model_measurements['Tree size (nodes)'])
#         live.log_metric("Tree size (leaves)",efdt.get_model_measurements['Tree size (leaves)'])
#         live.next_step()



# Display results

# print ('Execution time : ', elapsed, 'ms')
# print('{} samples analyzed.'.format(n_samples))

# print(f"Tree depth : {efdt.get_model_measurements['Tree depth']}")
# print(f"Number of nodes : {efdt.get_model_measurements['Tree size (nodes)']}")
# print(f"Number of leaves : {efdt.get_model_measurements['Tree size (leaves)']}")

# print('Extremely Fast Decision Tree accuracy: {}'.format(correct_cnt / n_test_samples))
# target_names = ['non diabetes', 'diabetes']
# print(classification_report(y_true, y_predicted, target_names=target_names))


#--------------------------------------------------------------------------------------
# Very Fast Decision Tree

# # Setting up a data stream
# training_stream = FileStream('training_set.csv')
# test_stream = FileStream('test.csv')

# # Setup Hoeffding Tree estimator
# ht = HoeffdingTreeClassifier()

# start = time.time()

# # Setup variables to control loop and track performance
# n_samples = 0
# correct_cnt = 0
# max_samples = 20000

# # Train the estimator with the samples provided by the data stream
# while n_samples < max_samples and training_stream.has_more_samples():
#     X, y = training_stream.next_sample()
#     ht.partial_fit(X, y)
#     n_samples += 1

# n_test_samples = 0

# # Test the estimator with the samples provided by the data stream
# y_true = []
# y_predicted = []

# while n_test_samples < max_samples and test_stream.has_more_samples(): 
#   X, y = test_stream.next_sample()
#   y_pred = ht.predict(X)
#   y_true.append(y)
#   y_predicted.append(y_pred)
#   if y[0] == y_pred[0]:
#       correct_cnt += 1
#   n_test_samples += 1

# end = time.time()
# elapsed = end - start

# # Display results

# print ('Execution time : ', elapsed, 'ms')
# print('{} samples analyzed.'.format(n_samples))

# print(f"Tree depth : {ht.get_model_measurements['Tree depth']}")
# print(f"Number of nodes : {ht.get_model_measurements['Tree size (nodes)']}")
# print(f"Number of leaves : {ht.get_model_measurements['Tree size (leaves)']}")

# print('Very Fast Tree accuracy: {}'.format(correct_cnt / n_test_samples))
# target_names = ['non diabetes', 'diabetes']
# print(classification_report(y_true, y_predicted, target_names=target_names))


# #--------------------------------------------------------------------------------------
# # Concept-adaptating Very Fast Decision Tree

# # Setting up a data stream
# training_stream = FileStream('training_set.csv')
# test_stream = FileStream('test.csv')

# # Setup Hoeffding Tree estimator
# cvfdt = HoeffdingAdaptiveTreeClassifier()

# start = time.time()

# # Setup variables to control loop and track performance
# n_samples = 0
# correct_cnt = 0
# max_samples = 20000

# # Train the estimator with the samples provided by the data stream
# while n_samples < max_samples and training_stream.has_more_samples():
#     X, y = training_stream.next_sample()
#     cvfdt.partial_fit(X, y)
#     n_samples += 1

# n_test_samples = 0

# # Test the estimator with the samples provided by the data stream
# y_true = []
# y_predicted = []

# while n_test_samples < max_samples and test_stream.has_more_samples(): 
#   X, y = test_stream.next_sample()
#   y_pred = cvfdt.predict(X)
#   y_true.append(y)
#   y_predicted.append(y_pred)
#   if y[0] == y_pred[0]:
#       correct_cnt += 1
#   n_test_samples += 1

# end = time.time()
# elapsed = end - start

# # Display results

# print ('Execution time : ', elapsed, 'ms')
# print('{} samples analyzed.'.format(n_samples))

# print(f"Tree depth : {cvfdt.get_model_measurements['Tree depth']}")
# print(f"Number of nodes : {cvfdt.get_model_measurements['Tree size (nodes)']}")
# print(f"Number of leaves : {cvfdt.get_model_measurements['Tree size (leaves)']}")

# print('Concept-adaptating Very Fast Tree accuracy: {}'.format(correct_cnt / n_test_samples))
# target_names = ['non diabetes', 'diabetes']
# print(classification_report(y_true, y_predicted, target_names=target_names))

# #--------------------------------------------------------------------------------------
# Decision Tree

with open ('data/prepared/training_set.csv', 'r', encoding="utf-8") as f:
  reader = csv.reader(f)
  training_data = list(reader)

with open ('data/prepared/test.csv', 'r', encoding="utf-8") as f:
  reader = csv.reader(f)
  test_data = list(reader)

#training set
training_outcome = [i[-1] for i in training_data]
training_without_outcome = [i[:-1] for i in training_data]

#test set
test_expected_outcome = [i[-1] for i in test_data]
test_without_outcome = [i[:-1] for i in test_data]

# Créer un classificateur d'arbre de décision
dt = DecisionTreeClassifier()
start = time.time()

with Live() as live:

    for i in range(10): 

        dt.fit(training_without_outcome, training_outcome)
        test_predicted_outcome = dt.predict(test_without_outcome)

        report = classification_report(test_expected_outcome, test_predicted_outcome, output_dict=True)
        precision_avg = report['macro avg']['precision']
        recall_avg = report['macro avg']['recall']
        f1_avg = report['macro avg']['f1-score']

        accuracy = report['accuracy']

        live.log_metric("accuracy",accuracy)
        live.log_metric("precision_avg",precision_avg)
        live.log_metric("recall_avg",recall_avg)
        live.log_metric("f1_avg",f1_avg)
        live.next_step()

end = time.time()
elapsed = end - start

# print ('Execution time : ', elapsed, 'ms')

# print(f"Tree depth : {dt.get_depth()}")
# print(f"Number of nodes : {dt.get_n_leaves()}")
# print(f"Number of leaves : {dt.tree_.node_count}")

# target_names = ['non diabetes', 'diabetes']
# print(classification_report(test_expected_outcome, test_predicted_outcome, target_names=target_names))

