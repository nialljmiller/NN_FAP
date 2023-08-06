# NN_FAP: Neural Network False Alarm Probability for Periodicity Verification

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/NN_FAP.svg)](https://badge.fury.io/py/NN_FAP)
[![Documentation Status](https://readthedocs.org/projects/nn-fap/badge/?version=latest)](https://nn-fap.readthedocs.io/en/latest/?badge=latest)

## Introduction

NN_FAP is a Python package designed to provide a simple and efficient way to verify the periodicity present in time-series astronomical data using a machine learning-based technique. This package utilizes a Recurrant Neural Network (RNN) to calculate the False Alarm Probability (FAP) directly from the phase-folded light curve. This method can be seen as a RNN approach to PDM & Conditional Entropy. We are searching for any "signal" in the phase folded light curve (or any x,y data...)


## Installation

To use the NN_FAP package, you can clone the repository from GitHub and install it locally. Here's how:

```bash
git clone https://github.com/your_username/NN_FAP.git
cd NN_FAP
pip install -e .
```

By running the above commands, the package will be installed in editable mode (`-e`), allowing you to modify the source code and see the changes immediately without reinstalling.

Alternatively, you can download the source code from the GitHub repository and use it directly in your project. Please ensure that you have the necessary dependencies installed before running the package.

Once the package is fully developed and ready for public distribution, you can consider packaging it and publishing it on PyPI to make it easily installable via `pip`.

## Usage

To perform the periodicity verification using NN_FAP, you can use the `inference` function provided in the package:

If the model is stored in the default path then:

```python
import NN_FAP

# Assuming you have your time-series data in `time` and `mag` arrays
# Example: time = [t1, t2, t3, ...], mag = [m1, m2, m3, ...]

# Provide the period guess for phase-folding
period_guess = 1.23456

# Call the inference function to get the False Alarm Probability
FAP = NN_FAP.inference(period_guess, mag, time)

print(f"False Alarm Probability: {FAP}")
```

for using a custom path with the model:

```python
import NN_FAP

# Assuming you have your time-series data in `time` and `mag` arrays
# Example: time = [t1, t2, t3, ...], mag = [m1, m2, m3, ...]

# Provide the period guess for phase-folding
period_guess = 1.23456

# Load the pre-trained model and KNN classifier using `get_model`
model_path = '/path/to/your/model_directory/'

knn, model = NN_FAP.get_model(model_path=model_path)

# Call the inference function with the loaded model and KNN
FAP = NN_FAP.inference(period_guess, mag, time, knn=knn, model=model)

print(f"False Alarm Probability: {FAP}")
```

for just passing your own model and KNN:

```python
import NN_FAP
from tensorflow.keras.models import model_from_json
from sklearn import neighbors

# Assuming you have your time-series data in `time` and `mag` arrays
# Example: time = [t1, t2, t3, ...], mag = [m1, m2, m3, ...]

# Provide the period guess for phase-folding
period_guess = 1.23456

# Load the pre-trained model
model_path = '/path/to/your/model_directory/'
json_file = open(model_path + '_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into the new model
loaded_model.load_weights(model_path + '_model.h5')
# Load model history
history = np.load(model_path + '_model_history.npy', allow_pickle=True).item()

# Load the KNN classifier
N = 200
knn_N = int(N / 20)
knn = neighbors.KNeighborsRegressor(knn_N, weights='distance')

# Call the inference function with the manually added model and KNN
FAP = NN_FAP.inference(period_guess, mag, time, knn=knn, model=loaded_model)

print(f"False Alarm Probability: {FAP}")
```


You can provide your own pre-trained model and k-Nearest Neighbors (KNN) classifier using the optional parameters `model` and `knn`. If not provided, the function will use the default model and KNN classifier.

## Documentation

%For detailed usage instructions, examples, and API reference, please refer to the [documentation](https://nn-fap.readthedocs.io/en/latest/).

## Citation

If you find this package useful in your research, please consider citing our paper:

[Paper Title Here](link_to_your_paper)

## Contributing

We welcome contributions to improve this package! If you find any issues or have suggestions for new features, feel free to open an issue or submit a pull request on [GitHub](https://github.com/your_username/NN_FAP).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

