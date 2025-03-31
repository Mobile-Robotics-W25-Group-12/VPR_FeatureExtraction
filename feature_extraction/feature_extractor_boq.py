import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data

from typing import List
import numpy as np
from tqdm.auto import tqdm


class BoQImageDataset(data.Dataset):
    def __init__(self, imgs, image_size):
        super().__init__()
        self.mytransform = self.input_transform(image_size)
        self.images = imgs

    def __getitem__(self, index):
        img = self.images[index]
        img = self.mytransform(img)
        return img, index

    def __len__(self):
        return len(self.images)

    @staticmethod
    def input_transform(image_size):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class BoQFeatureExtractor(torch.nn.Module):
    def __init__(self, backbone_name="resnet50"):
        super().__init__()

        self.backbone_name = backbone_name

        # Set image size and output dimension based on backbone
        if backbone_name == "resnet50":
            self.image_size = (384, 384)
            self.dim = 16384
        elif backbone_name == "dinov2":
            self.image_size = (322, 322)
            self.dim = 12288
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Set up device
        if torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("Using MPS")
            self.device = torch.device("mps")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        # Load model
        print(f"Loading BoQ model with {backbone_name} backbone...")
        self.model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name=backbone_name,
            output_dim=self.dim,
        )
        self.model = self.model.to(self.device)

    # Update the compute_features method in the BoQFeatureExtractor class to handle tuple outputs
    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        img_set = BoQImageDataset(imgs, self.image_size)
        test_data_loader = DataLoader(
            dataset=img_set,
            num_workers=4,
            batch_size=4,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        self.model.eval()
        with torch.no_grad():
            global_feats = np.empty((len(img_set), self.dim), dtype=np.float32)
            for input_data, indices in tqdm(test_data_loader):
                indices_np = indices.numpy()
                input_data = input_data.to(self.device)
                output = self.model(input_data)

                # Handle the case where output is a tuple
                if isinstance(output, tuple):
                    # Assuming the first element contains the features
                    image_encoding = output[0]
                else:
                    image_encoding = output

                global_feats[indices_np, :] = image_encoding.cpu().numpy()

        return global_feats
