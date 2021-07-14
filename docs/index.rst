Individual project : Deep learning proxy of reservoir simulation
=====================================================

Synopsis:
---------

Forecasting subsurface flow is highly important for oil and gas reservoir management and oilfield development optimization. It is indispensable to have a precise and quick model of multiphase flow through porous media to solve these problems. This typically requires thousands of expensive flow simulations.
Artificial neural networks (ANNs) models have been used to create a surrogate model to forecast flow. This type of models can reduce the simulation time considerably and can be applied in oil and gas management and development problems.
In this project, our objective is to use a combination of a convolutional neural network (CNN) model (residual U-Net) with a model of recurrent neural network (RNN) which is long short-term memory (LSTM) to predict the 2-D system flow. 


Dataset
~~~~~~~~~~~~~~~~~~~
You can download dataset from below google drive [link](https://drive.google.com/drive/folders/16o2DW8kXaXbDh7FrD_63uHNi4N1ffEk0?usp=sharing)

The dataset includes 2250 training and 7500 testing images that will be used for training ANN models.

  Trainging dataset:
    The training dataset is 2250 of All_X.npy and All_Y.npy as permeability map as input data and their corresponding pressure and saturation maps as the targets. 
 Test dataset:
    The training dataset is 750 of All_X.npy and All_Y.npy as permeability map as input data and their corresponding pressure and saturation maps as the targets.

Models Comparsion and choosing
~~~~~~~~~~~~~~~~~~~
We trained several different models to compare them and find the best ones based on their performance. The models we trained included: Residual U-NET LSTM and so on.

The images of performance of different models on original datasets can be find [here] ("acse-4-x-ray-classification-inception/img/img/")
.

We used transfer learning and pretrained on models to train our models.

Based training and validation loss and accuracy and comparsion of models.

Hyperparameters search
~~~~~~~~~~~~~~~~~~~
In the model, grid search has been dont to find the optimal hyperparameters; hyperparameters which choose to be searched:
- Learning rate
- Optimizer
- Batch Size
- momentum

Data Preprocessing
~~~~~~~~~~~~~~~~~~~
Mean, standard deviation, Max and min of training, test and validation set were calcualted. Then pressure data standardized and normalized using these statistical values


Function API
============

.. automodule:: inception
  :members:
  :imported-members:
