from classifier.classifier import NNClassifier
from validator.validator import Validator

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



if __name__ == "__main__":
    # test small dataset
    s_dataset = load_dataset("data/small-test-dataset-2-2.txt")

    classifier = NNClassifier()
    validator = Validator()

    # using [3, 5, 7] but since index starts from 0, it is rather [2,4,6]
    feature_subset = [2, 4, 6]
    accuracy = validator.evaluate(s_dataset, classifier, feature_subset)

    print(f"Accuracy using features {feature_subset}: {accuracy*100:.2f}%")

    # test large dataset
    l_dataset = load_dataset("data/large-test-dataset-2.txt")

    #using [1, 15, 27] but since index starts from 0, it is rather [0, 14, 26]
    feature_subset = [0, 14, 26]
    accuracy = validator.evaluate(l_dataset, classifier, feature_subset)

    print(f"Accuracy using features {feature_subset}: {accuracy*100:.2f}%")
