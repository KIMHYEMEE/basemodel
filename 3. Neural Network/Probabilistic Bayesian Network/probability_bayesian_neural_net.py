# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:58:11 2021

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
num_epochs = 1000

# 2. Load data set

train_dataset, test_dataset = get_train_and_test_splits(train_size, dataset_size, batch_size)

# 3. Modeling

mse_loss = keras.losses.MeanSquaredError()

prob_bnn_model = create_probablistic_bnn_model(FEATURE_NAMES, train_size, hidden_units, prior, posterior)


# 4. Training

run_experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset, num_epochs)

# 5. Test

sample = 10
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[0]

prediction_distribution = prob_bnn_model(examples)
prediction_mean = prediction_distribution.mean().numpy().tolist()
prediction_stdv = prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
prediction_stdv = prediction_stdv.tolist()

for idx in range(sample):
    print(
        f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
        f"stddev: {round(prediction_stdv[idx][0], 2)}, "
        f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
        f" - Actual: {targets[idx]}"
    )

