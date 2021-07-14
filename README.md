# Deep learning Proxy of reservoir simulation

## Project description
The common modelling method for reservoir fluid flow model is to use a finite-difference (FD). Since FD uses computations on a discrete computational grid, and its precision is strongly related on a grid resolution. Being accurate, fluid flow simulation through porous media may be really time consuming (several hours for a single run) and we have to perform hundreds of thousands of run to calibrate the models using observation data and to correctly quantifying uncertainty.
/
In oil industry a fast and accurate model are required for three-phase flow simulation on a 3D computational grid. All conventional approaches are unable to solve this task properly: either they hardly produce solutions for spatial fields (pressure, saturation, etc.) or they can work only under nonrealistic greatly simplified settings. In this project. 
We propose to employ deep learning (DL) methods to create an efficient proxy of the 3D fluid flow simulator. The objective is to train a DL model that will take 3D cells of permeability and porosity values as an input and then produce 3D images of saturation and pressure as a function of time with given discretization.

## Dataset
You can download dataset from below google drive [link](https://drive.google.com/drive/folders/16o2DW8kXaXbDh7FrD_63uHNi4N1ffEk0?usp=sharing)

The dataset includes 2250 training and 7500 testing images that will be used for training ANN models.

  Trainging dataset:
    The training dataset is 2250 of All_X.npy and All_Y.npy as permeability map as input data and their corresponding pressure and saturation maps as the targets. 
 Test dataset:
    The training dataset is 750 of All_X.npy and All_Y.npy as permeability map as input data and their corresponding pressure and saturation maps as the targets.

## Models Comparsion and choosing

We trained different CNN models to compare them and find the best ones based on their performance. The models we trained included: Residual-U-Net and LSTM,.

The images of performance of different models on original datasets can be find under [images]("") directory. We used transfer learning on pretrained models to develop our models. Based on the comparison of the accuracy and loss of the training and validation data, the ... and ... models were idetified as the best.

## Model Properties 

A grid search was used to determine the optimal model hyperparameters; the hyperparameters which were chosen to be optimised were:
- Learning rate
- Optimizer
- Batch Size
- momentum

The results of the grid search is given in the tables which can be found in the [tables](https://github.com/acse-2020/acse-4-x-ray-classification-inception/tree/main/tables) directory:
        
## Data Preprocessing and Augmentation

Mean and standard deviation, Max and min of training, test and validation set were calcualted. Then pressure data standardized and normalized using these statistical values.
            

## Installation

It is recommended that you create a new virtual environment and install the required `Python` packages
within your virtual environment.

If you wish to create an `Anaconda` environment, an `environment.yml` file has
been provided from which you can create the `proxy-model` environment
by running
```bash
conda env create -f environment.yml
```
from within the base folder of this repository. If you wish to use some other virtual environment solution (or none at all),
when in your environment ensure that all requirements are satisfied via
```bash
pip install -r requirements.txt
```

Whichever solution you decide upon, the package `inception` should then be installed
by running
```bash
pip install -e .
```
from within the base folder of this repository.

**NB**: Update the libraries to their latest versions before training.


## Downloading dataset

To download the data

```python
from models.preprocessing import download_file_from_google_drive
```

To use the models module you first need to import it:

```python
import inception
```
Then you can use different modules including layers and unet_uae

## Unit testing

To run the unit test suite
```
python tests.py
```

## Documentation

To generate the documentation (in html format)
```
python -m sphinx docs html
```

See the `docs` directory for the preliminary documentation provided that you should add to.


## More information

For more information on the project specfication, see the python notebooks: `ResNet18model.ipynb`.

## Further investigation

- Investigating performance of model and choose model based training and testind run time.

## References

#### Model architecture

* Huang, Gao, Zhuang Liu, Kilian Q Weinberger, and Laurens van der Maaten. "Densely Connected Convolutional Networks." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. [link](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html), [arXiv:1608.06993](https://arxiv.org/abs/1608.06993), [Torch implementation](https://github.com/liuzhuang13/DenseNet)
* Xie, Saining, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. "Aggregated Residual Transformations for Deep Neural Networks." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. [link](http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html), [arXiv:1611.05431](https://arxiv.org/abs/1611.05431), [Torch implementation](https://github.com/facebookresearch/ResNeXt)
* Huang, Gao, Zhuang Liu, Geoff Pleiss, Laurens van der Maaten, and Kilian Q. Weinberger. "Convolutional Networks with Dense Connectivity." IEEE transactions on pattern analysis and machine intelligence (2019). [arXiv:2001.02394](https://arxiv.org/abs/2001.02394)

