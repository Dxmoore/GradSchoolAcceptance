import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import uniform, randint

####Data Loading and Observing###

#Load in data
dataset = pd.read_csv('admissions_data.csv')

#view a sample of the data
dataset.head()
dataset.describe()



###Data Preprocessing###

#drop the serial no  column as this has no impact on results
dataset.drop(['Serial No.'], axis = 1) 

#create "labels" based on chance admit column
labels = dataset.iloc[:,-1]

#create features based on all other columns 
features = dataset.iloc[:,0:-1]
#convert categorical columns
features = pd.get_dummies(features)


#split data

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state=23)

#standardize numerical data
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
 
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

#exchange features data with scaled data
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.fit_transform(features_test)


## Define the model function

def create_model(num_epochs=200, batch_size=2): #uses default epoch and batch size values 
    model = Sequential(name="my_model")
    input = layers.InputLayer(input_shape=(features.shape[1],))
    model.add(input)
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(loss='mse', metrics=['mae'], optimizer='adam')
    return model

# Define the parameter grid for randomized search
param_grid = {
    'batch_size': randint(1, 10),
    'num_epochs': randint(100, 500),

}

# Create the KerasRegressor wrapper
model_wrapper = KerasRegressor(build_fn=create_model, verbose=0)

# Perform randomized search with early stopping
random_search = RandomizedSearchCV(
    model_wrapper,
    param_distributions=param_grid,
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    n_iter=12,
    cv=3
)

# Fit the randomized search to the data
random_search.fit(features_train_scaled, labels_train)

# Get the best parameters found by randomized search
best_params = random_search.best_params_

# Create the best model using the best parameters
best_model = create_model(num_epochs=best_params['num_epochs'], batch_size=best_params['batch_size'])

# Train the best model
best_model.fit(
    features_train_scaled,
    labels_train,
    epochs=best_params['num_epochs'],
    batch_size=best_params['batch_size'],
    validation_split= .2
    
)

# Evaluate the best model
loss, mae = best_model.evaluate(features_test_scaled, labels_test)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", loss)
print("Mean Absolute Error (MAE):", mae)

# Make predictions on the test set using the best model
predictions = best_model.predict(features_test_scaled)

# Calculate R-squared score
r2 = r2_score(labels_test, predictions)
print("R-squared score:", r2)

# Print the best parameters found by randomized search
print("Best Parameters:", best_params)


# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below
# Train the best model
history = best_model.fit(
    features_train_scaled,
    labels_train,
    epochs=best_params['num_epochs'],
    batch_size=best_params['batch_size'],
    validation_split=.2,
)

# Extract loss and MAE values from the history object
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(loss) + 1)

# Plot loss per epoch
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot MAE per epoch
plt.figure(figsize=(8, 6))
plt.plot(epochs, mae, 'b-', label='Training MAE')
plt.plot(epochs, val_mae, 'r-', label='Validation MAE')
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
