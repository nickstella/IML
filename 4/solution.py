# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

INPUT_SIZE = 1000
HIDDEN_LAYERS_SIZES = [400, 128, 50, 16]
OUTPUT_SIZE = 1
NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 0.0005
BATCH_SIZE = 256

VALIDATION = False

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, input_size, hidden_layers_sizes, output_size):
        """
        The constructor of the model.
        """
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_layers_sizes)):
            layers.append(nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers_sizes[-1], output_size))

        # Sequential model
        self.model_cut = nn.Sequential(*layers[:-1])
        self.model_whole = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        x = self.model_whole(x)
        return x

    def get_features(self, x):
        x = self.model_cut(x)
        return x
    
def make_feature_extractor(x, y, batch_size=BATCH_SIZE, validation=False, eval_size=10000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]

    if validation:
        x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
        x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
        y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    else:
        x_tr, y_tr = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    # model declaration
    model = Net(input_size=INPUT_SIZE, hidden_layers_sizes=HIDDEN_LAYERS_SIZES, output_size=OUTPUT_SIZE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUMBER_OF_EPOCHS):

        # Training
        model.train()
        train_loss = 0.0
        for i in range(0, len(x_tr), batch_size):
            X = x_tr[i : i+batch_size,]
            y = y_tr[i : i+batch_size].unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / (len(x_tr) // batch_size)

        if not validation:
            print(f"Epoch {epoch + 1}/{NUMBER_OF_EPOCHS}: Train Loss: {avg_train_loss:.4f}")
            continue

        # Validation
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(x_val), batch_size):
                X = x_tr[i: i + batch_size,]
                y = y_tr[i: i + batch_size].unsqueeze(1)
                outputs = model(X)
                loss = criterion(outputs, y)
                validation_loss += loss.item()
        avg_validation_loss = validation_loss / (len(x_val) // batch_size)

        print(f"Epoch {epoch + 1}/{NUMBER_OF_EPOCHS}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_validation_loss:.4f}")




    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """

        x_tens = torch.tensor(x, dtype=torch.float)

        model.eval()
        with torch.no_grad():
            y_pred = model.get_features(x_tens)

        return y_pred

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X).numpy()
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """

    model = RidgeCV(alphas = np.linspace(0.1, 200, num = 10000)) #default: leave one out cross-validation technique (efficient)

    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor = make_feature_extractor(x_pretrain, y_pretrain, validation=VALIDATION)
    feature_extractor(x_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()
    y_pred = np.zeros(x_test.shape[0])

    pipeline = Pipeline([('pretrainedfeatureselector', PretrainedFeatureClass(feature_extractor="pretrain")),
                         ('scale', StandardScaler()),
                         ('reg', regression_model)])
    pipeline.fit(x_train, y_train)
    score = pipeline.score(x_train, y_train)
    print(f"Score = {score}")
    y_pred = pipeline.predict(x_test.to_numpy())


    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")