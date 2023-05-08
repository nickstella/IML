# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
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
                              pin_memory=True, num_workers=8)

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

    print(f"Loading data...")

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
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / norms[:, np.newaxis]

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

    if train:
        assert X.shape == (119030, 6144) and y.shape == (119030, )
    else:
        pass

    print("Data correctly loaded.")

    return X, y


# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y=None, train=True, batch_size=64, shuffle=True, num_workers=8):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """

    print("Creating the data loader...")

    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)

    print("Data loader created.")
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
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 2)

    def forward_one_embedding(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.out(x))
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

        #distance = nn.PairwiseDistance(p=2, keepdim=True)
        #distAB = distance(outA, outB)
        #distAC = distance(outA, outC)

        #distances = torch.cat((distAB,distAC),dim=1)
        #out = F.relu(self.fc4(distances))
        #out = F.relu(self.fc5(out))
        #out = torch.sigmoid(self.out(out))

        return outA, outB, outC

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 2.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.mean(F.relu(distance_positive - distance_negative + self.margin))
        return losses

""" class ContrastiveLoss(nn.Module):
    def __init__(self, margin = 2.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1-label)*torch.pow(euclidean_distance, 2) + (label)*torch.pow(torch.clamp(self.margin - euclidean_distance, min = 0.0), 2))"""

def train_model(train_loader, validation=True):
    """
    The training procedure of the model; it accepts the training data, defines the model
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """

    print("Starting model training...")

    model = Net()
    model.train()
    model.to(device)
    n_epochs = 100
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part
    # of the training data as a validation split. After each epoch, compute the loss on the
    # validation split and print it out. This enables you to see how your model is performing
    # on the validation data before submitting the results on the server. After choosing the
    # best model, train it on the whole training data.


    validation_rate = 0.2
    triplet_num = len(train_loader.dataset)
    num_validation_data = int(np.floor(validation_rate * triplet_num)) if validation else 0
    shuffled_data = np.random.permutation(list(range(triplet_num)))

    train_sampler = SubsetRandomSampler(shuffled_data[num_validation_data:])
    real_train_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=train_sampler)

    if validation:
        validation_sampler = SubsetRandomSampler(shuffled_data[:num_validation_data])
        validation_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=validation_sampler)

    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []

    for epoch in tqdm(range(n_epochs), desc="Epoch"):

        if validation:
            model.train()

        for batch, [X, y] in tqdm(enumerate(real_train_loader),total=len(real_train_loader),leave=False,desc="Batch training"):
            y_pred = model.forward(X).squeeze()
            loss = criterion(y_pred, y.to(torch.float32))
            losses.append(loss)
            tqdm.write(f'epoch: {epoch:2} batch: {batch:2}   loss: {loss.item():10.8f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if validation:
            predictions = []
            model.eval()

            accuracies = []

            with torch.no_grad():
                for batch, [X, y] in tqdm(enumerate(validation_loader),total=len(validation_loader),leave=False,desc="Batch validation"):
                    X = X.to(device)
                    predicted = model.forward(X)
                    predicted_np = predicted.cpu().numpy().squeeze()
                    # Rounding the predictions to 0 or 1
                    predicted_np[predicted_np >= 0.5] = 0
                    predicted_np[predicted_np < 0.5] = 1
                    predictions.append(predicted_np)

                    y_np = y.cpu().numpy()

                    assert len(y_np) == len(predicted_np)

                    correct = np.count_nonzero(y_np==predicted_np)
                    accuracy = correct / len(y_np)

                    accuracies.append(accuracy)

                    tqdm.write(f"epoch: {epoch:2} batch: {batch:2}   prediction accuracy: {accuracy}")

            tqdm.write(f"Epoch {epoch:2} overall accuracy: {sum(accuracies)/len(accuracies)}")

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
