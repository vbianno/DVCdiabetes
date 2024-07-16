import csv
import random
import yaml

# Splitting the dataset into a training set and a test set

params = yaml.safe_load(open("params.yaml"))["prepare"]

#opening the dataset file
with open ('data/raw/diabetes_5050int.csv', 'r', encoding="utf-8") as f:
  reader = csv.reader(f)
  rows = list(reader)

# Splitting the rows of the dataset depending on the outcome
rows0 = [i for i in rows if i[-1] == '0']
rows1 = [i for i in rows if i[-1] == '1']

# Creating the test dataset with the first 1000 rows of each outcome
test_data = rows0[:params["size_train"]] + rows1[:params["size_train"]]
rows0,rows1 = rows0[params["size_train"]:] , rows1[params["size_train"]:]

training_data = rows0 + rows1

# Writing the training set in a file 
with open('data/prepared/training_set.csv', 'w') as f:
    writer = csv.writer(f)
    random.shuffle(training_data)
    writer.writerows(training_data)

# Writing the test set in a file
with open('data/prepared/test.csv', 'w') as f:
    writer = csv.writer(f)
    random.shuffle(test_data)
    writer.writerows(test_data)