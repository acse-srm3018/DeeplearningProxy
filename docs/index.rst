Project 4.3: X-Ray classification
=====================================================

Synopsis:
---------

The goal of this competition is to classify a dataset of X-Ray lung images into 4 classes (below you will find a detailed description of the data formats you will use). The 4 classes are:

- Covid-19: lung images of Covid-19 patients.
- Lung opacity (ground-glass opacity): images of lungs showing areas of hazy opacification or increased attenuation (CT) due to air displacement by fluid, airway collapse, fibrosis, or a neoplastic process.
- Pneumonia: lung images of viral pneumonia patients.
- Normal: lung images of healthy patients.

Target keys for the submission files:

{
"covid":0,
"lung_opacity":1,
"pneumonia":2,
"normal":3
}

Important Note:

The evaluation metric for the submissions will be the AUC-ROC. Make sure you use this metric when you design and train your classifiers. Keep in mind that the leaderboard scores are based on this metric and NOT on accuracy scores. Here, you will find a link with instructions on how to compute the AUC-ROC metric:
https://www.kaggle.com/nkitgupta/evaluation-metrics-for-multi-class-classification

Data
~~~~~~~~~~~~~~~~~~~

In the Data tab, you will find:

- Trainging dataset:
The training dataset is organised in 4 folders. Each folder name contains the name of one of the 4 classes. Inside each folder you will find images corresponding to the class of the folder name.
- Test dataset:
In the test dataset folder you will find 950 images unlabeled (of course).
submission_sample.csv: this is a file with dummy values that shows the right format of the submission file. Your submissions should have this format.

Task
~~~~~~~~~~~~~~~~~~~

Your goal is to correctly label all the images in the test dataset.

Models Comparsion and choosing
~~~~~~~~~~~~~~~~~~~
We trained several different models to compare them and find the best ones based on their performance. The models we trained included: Resnet18, Resnet50, VGG19, GoogleNet, Mobilenet-v2, inception-v2 and inceptionV3, Alexnet and so on.

The images of performance of different models on original datasets can be find [here] ("acse-4-x-ray-classification-inception/img/img/")
.

We used transfer learning and pretrained on models to train our models.

Based training and validation loss and accuracy and comparsion of models, Resenet18 and Mobilenet as best models.

Hyperparameters search
~~~~~~~~~~~~~~~~~~~
In the model, grid search has been dont to find the optimal hyperparameters; hyperparameters which choose to be searched:
- Learning rate
- Optimizer
- Batch Size
- momentum

Optimal hyperparameters for *Resnet 18* :
batch size of 32, learning rate 0.001 and Adam optimizers have shown the best result

Optimal hyperparameters for *Mobilenet V2*:
batch size of 32, learning rate 0.01 and SGD optimizers have shown the best result

The results of grid search has been given in tables which can be find in following tables:
Data Preprocessing
~~~~~~~~~~~~~~~~~~~
Man and standart deviation of training, test and validation set are chosen.

Data Augmentation
~~~~~~~~~~~~~~~~~~~
We applied data augmentation methods in training set to reduce overfitting on image data via artificially enlarge the dataset using label-preserving transformations (Krizhevsky et al., 2012). Also, because the dataset is unbalanced, we applied augmentation to solve this problem. We choosed images randomly to do three augmentation. After doing each augmentation, we saved the augmented images, so there were images which had be done more than one augmentation.

We use

        torchvision.transforms.RandomRotation(15) 

in augmentation_rotation function that randomly rotate the images from -15 degrees to 15 degrees.

        transforms.ColorJitter(contrast=0.5)
            
in augmentation_contrast function that change the contrast of images.
            
Also, we built our own class AddGaussianNoise and methods to generate Gaussian noise. In augmentation_Gaussian_Noise funtion, we added this noise to images.

Function API
============

.. automodule:: inception
  :members:
  :imported-members:
