# Neural Network for Recidivism Prediction

This code implements a neural network model for predicting recidivism using the COMPAS dataset. The model is trained and evaluated on the dataset, and hyperparameter tuning is performed to find the best-performing model.

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- pandas
- matplotlib
- requests
- zipfile
- io

## Dataset
The COMPAS dataset is used for training and evaluation. It contains information about individuals, including their demographics, criminal history, and risk scores. The dataset is downloaded from the Stanford University website and extracted for further processing.

## Data Preprocessing
The code performs the following preprocessing steps on the dataset:
1. Selects relevant fields of interest.
2. Renames columns for better interpretability.
3. Removes records with missing scores.
4. Converts string values to numerical values.
5. One-hot encodes the "race" variable.
6. Converts the processed dataset from a pandas dataframe to a numpy array.

## Model Training and Evaluation
The code defines two neural network models: `nn_classifier` and `dropout_classifier`. These models are trained and evaluated using the training and test datasets.

### nn_classifier
This model is initialized with dropout and consists of several fully connected layers with ReLU activation. It uses the softmax activation function for the output layer and is compiled with the SGD optimizer.

### dropout_classifier
This model is similar to `nn_classifier` but includes dropout regularization in the input layer. It is also compiled with the SGD optimizer.

Both models are trained using the training dataset and evaluated using the test dataset. The evaluation results include the test loss and accuracy.

## Hyperparameter Tuning
The code includes a function `tune_hyperparams` that performs hyperparameter tuning. It iterates over different combinations of learning rates and regularization strengths, trains models with these hyperparameters, and evaluates their performance. The best-performing model and its corresponding hyperparameters are returned.

## Usage
To use this code, ensure that the required dependencies are installed. You can then run the code to train and evaluate the models on the COMPAS dataset. Hyperparameter tuning can be performed by calling the `tune_hyperparams` function.

Note: This code assumes that the COMPAS dataset is available at the specified URL and follows a specific format. If using a different dataset, modifications may be required.

Feel free to modify the code as needed for your specific use case.
