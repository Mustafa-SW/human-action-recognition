from torch.utils.data import DataLoader

from datasets.custom_dataset import ActionDataset

dataset = ActionDataset("data/sample_data.csv")

print("Dataset Size:", len(dataset))

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