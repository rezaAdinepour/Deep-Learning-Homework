import pandas as pd
from sklearn.model_selection import train_test_split


# Load the dataset
file_path = '../inputs/dataset_1.csv'
dataset = pd.read_csv(file_path)

# Display the dataset
dataset.head(), dataset.shape



# Number of subsets
num_subsets = 50

# Stratified splitting function
def stratified_split(data, num_splits):
    subsets = []
    for _ in range(num_splits):
        stratified_sample, _ = train_test_split(data, test_size=(1 - 1/num_splits), stratify=data['label'])
        subsets.append(stratified_sample)
    return subsets

# Perform stratified split
subsets = stratified_split(dataset, num_subsets)

# Verify the first subset
subsets[0].head(), len(subsets)

