```
This python scripts are written from this reference(bayesian neural networks, keras)

 - Ref: https://keras.io/examples/keras_recipes/bayesian_neural_networks/#experiment-3-probabilistic-bayesian-neural-network
```

## 0. Function

This file has all the functions used in other executing scripts; baseline, bayesian_neural_net, probability_bayesian_neural_net

- 이 파일은 각 모델에서 활용되는 함수를 포함하고 있음

## 1. baseline

This script uses a simple neural network model. It means the output of model is an exact value.

- 이 코드는 기본 모델을 출력시키는 코드로, 일반적인 신경망 모델에서 출력하는 특정한 값을 출력함 

## 2. bayesian_neural_net

This script uses a bayesian approach. So the output of model is a range of the prediction.

- 이 코드는 베이지안을 기반으로하며, 예측값의 범위를 출력함

## 3. probability_bayesian_neural_net

This script results a probability distribution. It means the outputs of model are mean, standard deviation, and confidence interval.

- 이 코드는 확률 분포를 출력함. 이에 따라 출력되는 값은 평균, 분산, 신뢰구간임

## ImportError: IProgress not found

In this case, you can solve by executing these lines at command.

```
pip install ipywidgets
```
```
jupyter nbextension enable --py widgetsnbextension
```