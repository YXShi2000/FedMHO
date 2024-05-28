import os

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Function to visualize and save images
def visualize_and_save_images(images, save_path):
    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    axs = axs.ravel()

    for i in range(64):
        img = transforms.ToPILImage()(images[i])
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(save_path)
    plt.show()
    plt.close('all')


def data_clean(dataset, device, from_pretrain_model=False, category=0, threshold=0.8):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Load pre-trained model (pre-trained on ImageNet)
    if from_pretrain_model:
        model = models.resnet18(pretrained=True)
        model = model.to(device)
        model.eval()

    # Extract features from the pre-trained model
    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in data_loader:
            if from_pretrain_model:
                images = torch.cat([images, images, images], dim=1)  # Convert grayscale to RGB
                images = images.to(device)
                batch_features = model(images)
                features.append(batch_features.cpu().numpy())
            else:
                features.append(images.view(images.size(0), -1).numpy())
            labels.extend(batch_labels.numpy())

    features = torch.from_numpy(np.concatenate(features))
    if from_pretrain_model:
        features = features.view(features.size(0), -1)

    # Use k-means to find the cluster centers
    num_clusters = 1  # You can adjust this based on your needs
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(features)

    # Get the cluster centers
    feature_center = torch.from_numpy(kmeans.cluster_centers_).float()

    # Calculate the distance of each feature from the center
    distances = torch.norm(features - feature_center, dim=1)
    # print(distances)

    # Sort distances and get indices of the closest 50%
    num_samples = len(dataset)
    num_selected_samples = int(threshold * num_samples)  # 50% of samples
    _, indices = torch.topk(distances, k=num_selected_samples, largest=False)

    # Filter the dataset based on the selected indices
    filtered_dataset = torch.utils.data.Subset(dataset, indices)

    # Output the number of samples before and after cleaning
    print(f"Number of samples before cleaning: {len(dataset)}")
    print(f"Number of samples after cleaning: {len(filtered_dataset)}")

    try:
        os.mkdir("example_images")
    except:
        pass
    data_loader = DataLoader(filtered_dataset, batch_size=64, shuffle=True)
    for batch in data_loader:
        # Assuming your images are in the first element of the batch tuple
        images = batch[0]
        # Visualize and save the images
        save_path = 'example_images/{}.png'.format(category)  # Change this to your desired save path
        visualize_and_save_images(images, save_path)
        break  # Only visualize the first batch for demonstration purposes

    plt.scatter(range(len(distances)), distances.numpy(), c=kmeans.labels_, cmap='viridis')
    plt.title('K-means Clustering Results')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance to Cluster Center')
    plt.savefig('example_images/visual{}.png'.format(category))
    plt.show()

    return filtered_dataset
