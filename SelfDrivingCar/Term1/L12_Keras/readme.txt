Install Keras
Software requirements:
TensorFlow
Python 3
Instructions
Create a new conda environment.
conda create -n keras python=3.5
source activate keras
Install TensorFlow.
conda install -c conda-forge tensorflow
If you want to use the GPU version, follow the instructions here
Install Jupyter.
conda install jupyter
Install Keras. You only need the latest stable version.
pip install keras
You might need to pip[3] install:
h5py
scipy
scikit-learn
Pillow
Behind-the-scenes, Keras uses either TensorFlow, or an alternative deep learning library called Theano. By default Keras uses TensorFlow. However, if this is not the case, create ~/.keras/keras.json to configure Keras to use TensorFlow.

{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "floatx": "float32"
}
Test

python (should open python 3.5)
import keras (should say “Using TensorFlow backend” without any errors)
Celebrate!