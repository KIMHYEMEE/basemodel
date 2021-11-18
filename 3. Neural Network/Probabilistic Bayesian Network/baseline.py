# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:52:40 2021

@author: USER
"""

import os

os.chdir('C:/GIT/basemodel/3. Neural Network/Probabilistic Bayesian Network')

from function import *

# 1. Parameters

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

dataset_size = 4898
batch_size = 256
train_size = int(dataset_size * 0.85)
hidden_units = [8, 8]
num_epochs = 100


# 2. Load data set

train_dataset, test_dataset = get_train_and_test_splits(train_size, dataset_size, batch_size)


# 3. Modeling

mse_loss = keras.losses.MeanSquaredError()
baseline_model = create_baseline_model(FEATURE_NAMES, hidden_units)

# 4. Training

run_experiment(baseline_model, mse_loss, train_dataset, test_dataset, num_epochs)


# 5. Test

sample = 10
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[0]

predicted = baseline_model(examples).numpy()

for idx in range(sample):
    print(f"Predicted: {round(float(predicted[idx][0]), 1)} - Actual: {targets[idx]}")