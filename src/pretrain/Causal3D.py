import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Causal3DDataset(Dataset):
    def __init__(self, images_dir, metadata_file, transform=None):
        """
        Args:
            images_dir (str): Directory with all the images.
            metadata_file (str): Path to the metadata CSV file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.metadata.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        metadata = self.metadata.iloc[idx, 1:].to_dict()

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'metadata': metadata}

        return sample

from torch.utils.data import DataLoader
from torchvision import transforms

# Define any transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add other transformations as needed
])

# Create the dataset
dataset = Causal3DDataset(images_dir='path/to/images', metadata_file='path/to/metadata.csv', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for i, sample in enumerate(dataloader):
    images = sample['image']
    metadata = sample['metadata']
    # Your processing code here

