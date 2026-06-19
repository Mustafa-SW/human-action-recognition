from torch.utils.data import DataLoader

from datasets.custom_dataset import ActionDataset

dataset = ActionDataset("data/sample_data.csv")

print("Dataset Size:", len(dataset))

# DataLoader is a built-in PyTorch utility. Its job is to handle batching, shuffling, 
# and managing CPU memory threads when feeding data to an AI model

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for features, labels in loader:

    print("Features:")
    print(features)

    print("Labels:")
    print(labels)

    break

# Why use break here?  Instead of letting the loop run forever 
# and printing thousands of rows to your console, 
# adding break allows you to inspect just one single batch to make sure 
# the shapes, dimensions, and data types match exactly