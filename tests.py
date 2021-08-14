### Unit test for some models including RR U-net and A RR U-net###
# import required libraries and packages
import unittest
import vae_uae as vae 
from inception.utils import download_file_from_google_drive, image_to_tensor, batchnorm
from pathlib import Path
import tensorflow as tf 
import os
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import unet_uae as vae_util
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model
from tensorflow.python.keras import backend as K
from keras.optimizers import Adam
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)


def make_predictions(model, test_path):
    """
    Testing to ensure model input and true values enter to model correctly.
    Parameters
    ----------
    model: Dict
        the model 
    test path: string
        the location of test file
    Returns
    -------
    correct_preds: float
        tensor including targets
    all_samples: float
        tensor including all_samples
    """

    correct_preds = 0
    all_samples = 0
    for _, _, filenames in os.walk(test_path):
        all_samples = len(filenames)
        for filename in filenames:
            X = batchnorm(image_to_tensor(test_path + filename)).to('cpu')
            pred = int(model(X).max(1)[1].item()) + 1
            label = int(filename.split('_')[0])
            correct_preds += 1 if pred == label else 0
    return correct_preds, all_samples

class TestModelOutputShape(unittest.TestCase):
    """
    Tests to ensure the output tensor has the correct shape.
    """
    def test_vae_net(self):
        """
        Testing vae_net model with test data
        ----------
        Returns
        -------
        True: boolean
        """
        model = inception_v2(pretrained=True).to('cpu')
        x = torch.ones(8, 3, 299, 299)
        y = model(x)
        self.assertEqual(y.shape, torch.Size([8, 4]))
        return True

class TestModelAccuracy(unittest.TestCase):
    """
    Testing to nsure about shaping of model output
    """
    def __init__(self, *args, **kwargs):
        """
        Testing inception_v2 model with test data
        ----------
        Returns
        -------
        True: boolean
        """
        super(TestModelAccuracy, self).__init__(*args, **kwargs)
        self.dir_path = os.path.dirname(os.path.realpath(__file__)) + '/test_images'
        test_dir = Path(self.dir_path)
        if not test_dir.is_dir():
            zip_path = self.dir_path + '.zip'
            print("Download test images")
            download_file_from_google_drive('1nILi82OjswYySH631dcLaHZyoVB4cNJS', zip_path)
            print("Extract test images")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dir_path)
            os.remove(zip_path)
        else:
            print('Test images are already downloaded')
        self.dir_path += '/test_images/'

    def test_vae_net(self):
        """
        Testing vae_net model
        """
        model = inception_v2(pretrained=True).to('cpu')
        correct_preds, all_samples = make_predictions(model, self.dir_path)
        self.assertGreater(correct_preds, 0.6 * all_samples)

if __name__ == '__main__':
    unittest.main()
