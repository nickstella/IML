
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract
    the embeddings.
    """

    print("Generating embeddings...")

    # The following transform preprocesses the images to make them compatible with the chosen model
    # Note 1: I have seen that most people resize to 256 and crop to 224 instead of resizing to 224
    #       directly, not sure why though.
    # Note 2: I found out that ToTensor already transforms the image pixel values to move them from
    #       the range [0,255] to [0.0,1.0]. Normalization makes more sense now: we are subtracting
    #       a mean and dividing std in [0.0,1.0]. The values for mean and std are well known and match
    #       ResNet50, or whatever model we are going to use.
    # TODO Nickstar: play with these below according to the pretrained model you choose
    train_transforms = transforms.Compose([transforms.Resize((256,256)),
                                           transforms.CenterCrop((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Images are retrieved and transformed using the transformation defined above
    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)

    # Images are loaded in an iterable DataLoader object
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=3)

    # Definition of a model for extraction of the embeddings
    # TODO Nickstar: play with different models
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    embeddings = []
    # TODO Nickstar: this is the size of each embedding. It will depend on the model you play with.
    embedding_size = 2048
    assert embedding_size == model.fc.in_features

    # Replacement of the last layer of the ResNet architecture with an identity layer
    model.fc = nn.Identity()

    # Images are fed to the model to retrieve their embeddings
    model.eval()

    # Each 'images' is a tensor of shape [64,3,224,224], that is, 64 images.
    for images, _ in train_loader:
        with torch.no_grad():
            embedding_batch = model(images)
        embeddings.append(embedding_batch.numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    print(f"Embeddings correctly generated. A numpy array of shape {embeddings.shape} will now be saved to disk.")
    print(embeddings)

    np.save('dataset/embeddings.npy', embeddings)



def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    i=0
    with open(file) as f:
        for line in f:
            i+=1
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')

    # Normalization of embeddings. Each embedding will now be a 2048-dimensional vector with norm 1.
    print("Prior to normalization:")
    print(embeddings)

    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / norms[:, np.newaxis]

    print("After normalization:")
    print(embeddings)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)

    # Shape of X for train set is (119030, 6144) = (59515*2, 2048*3)
    # Shape of y for train set is (119030, 1)
    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y=None, train=True, batch_size=64, shuffle=True, num_workers=8):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader


class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(2,1)

    def forward_one_embedding(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        A, B, C = torch.chunk(x,3,dim=1)
        outA = self.forward_one_embedding(A)
        outB = self.forward_one_embedding(B)
        outC = self.forward_one_embedding(B)

        distAB = torch.pairwise_distance(outA, outB, p=2)
        distAC = torch.pairwise_distance(outA, outC, p=2)

        distances = torch.cat([distAB,distAC])

        out = torch.sigmoid(self.out(distances))

        return out

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part
    # of the training data as a validation split. After each epoch, compute the loss on the
    # validation split and print it out. This enables you to see how your model is performing
    # on the validation data before submitting the results on the server. After choosing the
    # best model, train it on the whole training data.

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        for [X, y] in train_loader:
            pass
    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data

    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad():  # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if not os.path.exists('dataset/embeddings.npy'):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train=True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
