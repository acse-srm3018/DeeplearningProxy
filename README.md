# Deep Learning Proxy of Reservoir Simulation

## Project description
The common modelling method for modeling reservoir fluid flow is to use a finite-difference (FD). Since FD uses computations on a discrete computational grid, and its precision is depend on a grid resolution. Being accurate, fluid flow simulation through porous media may be really time consuming (several hours for a single run) and we have to perform hundreds of thousands of run to calibrate the models using observation data and to correctly quantifying uncertainty.
/n
In oil industry, a fast and accurate model are required for three-phase flow simulation on a 3D computational grid. Almost all conventional approaches are unable to solve this task properly: either they hardly produce solutions for spatial fields (pressure, saturation, etc.) or they can work only under nonrealistic greatly simplified settings. In this project. 
We propose to employ deep learning (DL) methods to create an efficient proxy of the 3D fluid flow simulator. The objective is to train a DL model that will take 3D cells of permeability and porosity values as an input and then produce 3D images of saturation and pressure as a function of time with given discretization.

## Dataset
You can download  pressure dataset from google drive [link1](https://drive.google.com/file/d/1ngjOd_SPuD5NVDLbc06CQyAxKdZaB6ve/view?usp=sharing), [link2](https://drive.google.com/file/d/1fYA8LaNqnSPlwjGFKD69mKKOnmbVY5o7/view?usp=sharing) and the saturation [link1](https://drive.google.com/file/d/1sheRgkImbdhLQdz-xL9YnzA9Aib4acCG/view?usp=sharing), [link2](https://drive.google.com/file/d/1uD8x6S8UTKDcdDkXYrHfeGM-EjmASkNp/view?usp=sharing) and [permeability](https://drive.google.com/file/d/1hB4HyjT4k1q0Dqq3l2i66VXVw_Nf-vLc/view?usp=sharing).


The dataset includes 2250 training and 750 testing images that will be used for training ANN models.

 Trainging dataset:
    The training dataset is 2250 of permeability.npy and pressure.npy and saturation.npy as permeability map as input data and their corresponding pressure and saturation maps as the     targets. 
 Test dataset:
    The training dataset is 750 of permeability.npy and pressure.npy and saturation.npy as permeability map as input data and their corresponding pressure and saturation maps as the targets.

## Models Comparsion and choosing

We trained different CNN models to compare them and find the best ones based on their performance. The models we trained are Recurrent Residual-U-Net (RR U-Net) same as what has been used in this [work](https://github.com/soloist96/2D-recurrent-R-U-Net-surrogate-for-dynamic-flows) and attention residual recurrent U-net (A RR U-Net).

The images of performance of different models on original datasets can be find under [images](https://github.com/acse-srm3018/DeeplearningProxy/tree/main/images) directory. We used transfer learning on pretrained models to develop our models. Based on the comparison of the accuracy and loss of the training and validation data, the ... and ... models were idetified as the best.

## Model Properties 

A grid search was used to determine the optimal model hyperparameters; the hyperparameters which were chosen to be optimised were:
- Learning rate
- Batch Size

The results of the grid search is given in the tables which can be found in the [tables](https://github.com/acse-2020/acse-4-x-ray-classification-inception/tree/main/tables) directory:
        
## Data Preprocessing

Mean and standard deviation, Max and min of training and validation set were calculated. Then pressure data standardized and normalized using these statistical values.
            

## Installation

It is recommended that you create a new virtual environment and install the required `Python` packages
within your virtual environment.

If you wish to create an `Anaconda` environment, an `environment.yml` file has
been provided from which you can create the `proxy-model` environment
by running
```bash
conda env create -f environment.yml
```
Within the base folder of this repository. If you wish to use some other virtual environment solution (or none at all),
when in your environment ensure that all requirements are satisfied via
```bash
pip install -r requirements.txt
```

Whichever solution you decide upon, the package `models` should then be installed
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
import models
```
Then you can use different modules including layers and unet_uae.

## Results

The animation of pressure results from simulator is shown below.

<img src="https://github.com/acse-srm3018/DeeplearningProxy/blob/main/animation/animation_pressure_true_simulator.gif" alt="Overview" width="60%" height="60%"></a>

and the attention map from Attention RR U-Net model for different time from 90 days after production until 1800 days.

<img src="https://github.com/acse-srm3018/DeeplearningProxy/blob/main/animation/animation_pressure_attention_pred.gif" alt="overview" width="60%" height="60%"></a>

and saturation map animation

<img src="https://github.com/acse-srm3018/DeeplearningProxy/blob/main/animation/animation_saturation.gif" alt="overview" width="60%" height="60%"></a>

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

Also the pdf file of report placed in [report](https://github.com/acse-srm3018/DeeplearningProxy/blob/main/SeyedehRahaMoosavi_ACSE9_FinalReport.pdf) file.


## More information

For more information on the project specfication, see the [notebooks](https://github.com/acse-srm3018/DeeplearningProxy/tree/main/Notebooks) directory.
For use the saved models with 10 time steps, you can use:
- RR U-Net for pressure model can be found in this [saved pressure RR U-Net model](https://drive.google.com/file/d/1KIqCAb0x7xoZmTSg0MlSYRPbXPSilWVF/view?usp=sharing).
- A RR U-Net for pressure model can be found in this [saved pressure A RR U-Net model](https://drive.google.com/file/d/1hgLPrv5yL9yn8MHt0BnQQQCBAXCRlRbv/view?usp=sharing).
- RR U-Net for saturation model can be found in this [saved saturation RR U-Net model](https://drive.google.com/file/d/1WWsKbKtlaxjGHTNZiAbLucjvrIiMgWSE/view?usp=sharing).
- A RR U-Net for saturation model can be found in this [saved  saturation A RR U-Net model](https://drive.google.com/file/d/1Ph8zCWKTWR2q339V44PINRVvrxkSGb4d/view?usp=sharing).


## Further investigation

- Investigating performance of model and choose model based training and test run time.

## References

#### Model architecture

* Heinrich, M.P., Stille, M. and Buzug, T.M., 2018. Residual U-net convolutional neural network architecture for low-dose CT denoising. Current Directions in Biomedical Engineering, 4(1), pp.297-300.
* Alom, M.Z., Yakopcic, C., Hasan, M., Taha, T.M. and Asari, V.K., 2019. Recurrent residual U-Net for medical image segmentation. Journal of Medical Imaging, 6(1), p.014006.
* Azad, R., Asadi-Aghbolaghi, M., Fathy, M. and Escalera, S., 2019. Bi-directional ConvLSTM U-Net with densley connected convolutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (pp. 0-0).
* Jin, Z. L., Liu, Y. & Durlofsky, L. J. (2019), ‘Deep-learning-based reduced-order modeling for subsurface flow simulation’, arXiv preprint arXiv:1906.03729.
