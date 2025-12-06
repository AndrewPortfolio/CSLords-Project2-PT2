from classifier.classifier import NNClassifier
from validator.validator import Validator
import time
import math
import pandas as pd

def load_dataset(path):
    data = pd.read_csv(path, header=None, sep=r"\s+")

    dataset = []
    for _, row in data.iterrows():
        instance = {
            "label": int(row[0]),         
            "features": list(row[1:].values)  
        }
        dataset.append(instance)
    return dataset

def normalize(dataset):
    # this will get means and standard deviation for each feature accross instances
    # then finally return normalized dataset
    if len(dataset) == 0:
        return []

    num_features = len(dataset[0]["features"])
    num_instances = len(dataset)

    # compute means
    feature_means = [0.0] * num_features
    for instance in dataset:
        for i in range(num_features):
            feature_means[i] += instance["features"][i]
    feature_means = [m / num_instances for m in feature_means]

    # compute standard deviations
    feature_variances = [0.0] * num_features
    for instance in dataset:
        for i in range(num_features):
            diff = instance["features"][i] - feature_means[i]
            feature_variances[i] += diff ** 2
    std_dev = [math.sqrt(v / num_instances) for v in feature_variances]

    # normalize features
    normalized_dataset = []
    for instance in dataset:
        norm_features = []
        for i in range(num_features):
            if std_dev[i] > 0:
                norm_val = (instance["features"][i] - feature_means[i]) / std_dev[i]
            else:
                norm_val = 0.0
            norm_features.append(norm_val)
        normalized_dataset.append({
            "label": instance["label"],
            "features": norm_features
        })

    return normalized_dataset


if __name__ == "__main__":
    # test small dataset
    s_dataset = load_dataset("data/small-test-dataset-2-2.txt")

    classifier = NNClassifier()
    validator = Validator()

    # using [3, 5, 7] but since index starts from 0, it is rather [2,4,6]
    print("\nGetting Accuracy for SMALL DATASET (w/o normalization)")
    feature_subset = [2, 4, 6]
    start = time.time()
    accuracy = validator.evaluate(s_dataset, classifier, feature_subset)
    end = time.time()

    print(f"Accuracy using features [3, 5, 7]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")

    # test large dataset
    l_dataset = load_dataset("data/large-test-dataset-2.txt")

    #using [1, 15, 27] but since index starts from 0, it is rather [0, 14, 26]
    print("\nGetting Accuracy for LARGE DATASET (w/o normalization)")
    feature_subset = [0, 14, 26]
    start = time.time()
    accuracy = validator.evaluate(l_dataset, classifier, feature_subset)
    end = time.time()

    print(f"Accuracy using features [1, 15, 27]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")


    # after normalization
    print("\nApplying Normalization to each dataset... ", end='')
    s_dataset = normalize(s_dataset)
    l_dataset = normalize(l_dataset)
    print("Done!")

    print("\nGetting Accuracy for SMALL DATASET (w/ normalization)")
    feature_subset = [2, 4, 6]
    start = time.time()
    accuracy = validator.evaluate(s_dataset, classifier, feature_subset)
    end = time.time()
    print(f"Accuracy using features [3, 5, 7]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")

    print("\nGetting Accuracy for LARGE DATASET (w/ normalization)")
    feature_subset = [0, 14, 26]
    start = time.time()
    accuracy = validator.evaluate(l_dataset, classifier, feature_subset)
    end = time.time()

    print(f"Accuracy using features [1, 15, 27]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")
    print("\n")




    # ADDED TWO RANDOM TEST CASES TO SEE HOW IT WORKS!!

    #using [5, 20, 31] but since index starts from 0, it is rather [4, 19, 30]
    # feature_subset = [4, 19, 30]
    # start = time.time()
    # accuracy = validator.evaluate(l_dataset, classifier, feature_subset)
    # end = time.time()

    # print(f"Accuracy using features [5, 20, 31]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")
    
    # #using [10, 25, 36] but since index starts from 0, it is rather [9, 24, 35]
    # feature_subset = [9, 24, 35]
    # start = time.time()
    # accuracy = validator.evaluate(l_dataset, classifier, feature_subset)
    # end = time.time()

    # print(f"Accuracy using features [10, 25, 36]: {round(accuracy*100,2)}% -- took {round(end-start, 5)}seconds")
