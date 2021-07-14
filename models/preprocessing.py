"""
Preproceesing Data including Loading data.

normalized and standardized or both of them.
"""

import numpy as np

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/drive/folders/1A_mWXlDPWwfkjpD6I3faBn3yz_z2KHOh?usp=sharing"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 
    
    
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

                
def check_existence_and_download(id, destination):
    file_path = Path(destination)
    if file_path.exists():
        print(F"{destination} is already downloaded")
    else:
        print(F"Downloading {destination}")
        download_file_from_google_drive(id, destination)
        print(F"{destination} downloaded")   
        
        
def load_data(path):
    """
    Load datasets in format .NPY.

    Parameter:
    ----------
    path : string
        The absolute path of where data saved in local system
    Return:
    ----------
    loaded_data : ndarray
        The data which was loaded
    """
    loaded_data = np.load(path)
    return loaded_data


# Two common methods for feature scaling is :
# Normalization & Standardaisation
def normalize(data):
    """
    Use for Max-Min Normalization (Min-Max scaling) by re-scaling.

    features with a distribution value between 0 and 1. For every feature,
    the minimum value of that feature gets transformed into 0,
    and the maximum value gets transformed into 1

    Parameter:
    ----------
    data : ndarray
        The numpy array which we want to normalize

    Return:
    ----------
    norm_data : ndarray
        The normalized data which transformed into 0 and 1
    """
    max_p = np.max(data[:, :, :, :])
    min_p = np.min(data[:, :, :, :])
    norm_data = (data - min_p)/(max_p - min_p)
    return norm_data


def standardize(data):
    """
    Use for rescaling faetures to ensure the mean.

    and the standard deviation to be 0 and 1, respectively.

    Parameter:
    ----------
    data : ndarray
        The numpy array which we want to normalize
    Return:
    ----------
    data : ndarray
        The standardized data which the mean
    and the standard deviation to be 0 and 1
    """
    data_mean = np.mean(data[:, :, :, :], axis=0, keepdims=True)
    data_std = np.std(data[:, :, :, :], axis=0, keepdims=True)
    std_data = (data - data_mean)/(data_std)
    return std_data


def standardizeNormalize(data):
    """
    Use for rescaling faetures to ensure the mean.

    and the standard deviation to be 0 and 1, and data will be between
    0 and 1, respectively.

    Parameter:
    ----------
    data : ndarray
        The numpy array which we want to normalize

    Return:
    ----------
    data : ndarray
        The standardized data which the mean
    and the standard deviation to be 0 and 1
    """
    data_mean = np.mean(data[:, :, :, :], axis=0, keepdims=True)
    data = (data - data_mean)
    max_p = np.max(data[:, :, :, :])
    min_p = np.min(data[:, :, :, :])
    stdnorm_data = ((data - min_p)/(max_p - min_p)) - 0.5
    return stdnorm_data
