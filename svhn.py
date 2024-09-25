from torchvision import datasets

# Download and save SVHN dataset locally to './data/svhn/' folder
train_dataset = datasets.SVHN(root='/workspace', split='train', download=True)
test_dataset = datasets.SVHN(root='/workspace', split='test', download=True)

print("SVHN dataset downloaded and saved to ./data/svhn/")