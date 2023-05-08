
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
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VALIDATION = True

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

    #print("Prior to normalization:")
    #print(embeddings)

    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / norms[:, np.newaxis]

    #print("After normalization:")
    #print(embeddings)

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

def split_for_validation(X: np.ndarray, y: np.ndarray, rate=0.2):
    X_tmp = np.concatenate([X[::2], X[1::2]], axis=1)
    assert X_tmp.shape == (119030//2, 2048*6)
    y_tmp = np.concatenate([y[::2,np.newaxis], y[1::2,np.newaxis]], axis=1)
    assert y_tmp.shape == (119030//2, 2)

    X_y_tmp = np.concatenate([X_tmp,y_tmp], axis=1)
    assert X_y_tmp.shape == (119030//2, 2048*6+2)

    np.random.shuffle(X_y_tmp)

    X_tmp_chunked = X_y_tmp[:,:-2]
    assert X_tmp_chunked.shape == (119030 // 2, 2048 * 6)
    y_final_chunked = X_y_tmp[:, -2:]
    y_final = np.concatenate([y_final_chunked[:,0], y_final_chunked[:,1]], axis=0).squeeze()
    assert  y_final.shape == (119030, )

    n_rows = X_tmp_chunked.shape[0]
    n_val = int(np.floor(rate * n_rows))

    X_train_chunked, X_val_chunked = X_tmp_chunked[n_val:,:], X_tmp_chunked[:n_val,:]
    print(X_train_chunked.shape, X_val_chunked.shape)

    X_train = np.concatenate([X_train_chunked[:,:2048*3], X_train_chunked[:,2048*3:]], axis=0)
    X_val = np.concatenate([X_val_chunked[:,:2048*3], X_val_chunked[:,2048*3:]], axis=0)
    y_train = y_final[2*n_val:]
    y_val = y_final[:2*n_val]

    print(X_train, y_train, X_val, y_val)

    return X_train, y_train, X_val, y_val


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
        self.fc4 = nn.Linear(2,32)
        self.fc5 = nn.Linear(32, 16)
        self.out = nn.Linear(16,1)

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

        distance = nn.PairwiseDistance(p=2, keepdim=True)
        distAB = distance(outA, outB)
        distAC = distance(outA, outC)

        distances = torch.cat((distAB,distAC),dim=1)

        output = F.relu(self.fc4(distances))
        output = F.relu(self.fc5(output))
        output = self.out(output)
        output = torch.sigmoid(output)

        return output

def train_model(train_loader, val_loader, validation=False):
    """
    The training procedure of the model; it accepts the training data, defines the model
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """

    print("Starting model training...")

    if val_loader is None and validation:
        raise Exception("Cannot validate without appropriate loader!")

    model = Net()
    model.train()
    model.to(device)
    n_epochs = 1
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part
    # of the training data as a validation split. After each epoch, compute the loss on the
    # validation split and print it out. This enables you to see how your model is performing
    # on the validation data before submitting the results on the server. After choosing the
    # best model, train it on the whole training data.

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    losses = []
    accuracies = []


    for epoch in tqdm(range(n_epochs), desc="Epoch"):

        epoch_losses = []

        if validation:
            model.train()

        for batch, [X, y] in tqdm(enumerate(train_loader),total=len(train_loader),leave=False,desc="Batch training"):
            y_pred = model.forward(X).squeeze()
            loss = criterion(y_pred, y.to(torch.float32))
            epoch_losses.append(loss.item())
            tqdm.write(f'epoch: {epoch:2} batch: {batch:2}   loss: {loss.item():10.8f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = sum(epoch_losses)/len(epoch_losses)
        losses.append(avg_loss)

        if validation:
            predictions = []
            model.eval()

            epoch_accuracies = []

            with torch.no_grad():
                for batch, [X, y] in tqdm(enumerate(val_loader),total=len(val_loader),leave=False,desc="Batch validation"):
                    X = X.to(device)
                    predicted = model.forward(X)
                    predicted_np = predicted.cpu().numpy().squeeze()
                    # Rounding the predictions to 0 or 1
                    predicted_np[predicted_np >= 0.5] = 1
                    predicted_np[predicted_np < 0.5] = 0
                    predictions.append(predicted_np)

                    y_np = y.cpu().numpy()

                    #print(y_np)
                    #print(y_np.shape)
                    #print(predicted_np)
                    #print(predicted_np.shape)
                    #print(y_np==predicted_np)
                    #print()
                    #print(np.count_nonzero(y_np==predicted_np))


                    assert len(y_np) == len(predicted_np)

                    correct = np.count_nonzero(y_np==predicted_np)
                    accuracy = correct / len(y_np)

                    epoch_accuracies.append(accuracy)

                    tqdm.write(f"epoch: {epoch:2} batch: {batch:2}   prediction accuracy: {accuracy}")

            avg_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)
            accuracies.append(avg_accuracy)
            tqdm.write(f"Epoch {epoch:2} overall accuracy: {avg_accuracy}")


    try:
        fig, ax = plt.subplots(2)
        if validation:
            ax[1].set_title('Accuracy in epochs')
            ax[1].plot([i for i in range(n_epochs)], [item for item in accuracies])
            print(accuracies)

        ax[0].set_title('Loss in epochs')
        ax[0].plot([i for i in range(n_epochs)], [item for item in losses])

        plt.show()
    except:
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
    val_loader = None

    if VALIDATION:
        X_train, y_train, X_val, y_val = split_for_validation(X, y)
        train_loader = create_loader_from_np(X_train, y_train, train=True, batch_size=64)
        val_loader = create_loader_from_np(X_val, y_val, train=True, batch_size=64)
    else:
        train_loader = create_loader_from_np(X, y, train=True, batch_size=64)

    test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader, val_loader, validation=VALIDATION)

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
