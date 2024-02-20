# Exercises for the Deep Learning course held at FEI, VSB-TU Ostrava.

Course information may be found [here](https://homel.vsb.cz/~pla06/).

You can find more details about the course at [my homel](https://homel.vsb.cz/~svo0175/).

Feel free to [contact me](radek.svoboda@vsb.cz) if you have any questions or want to discuss any topic from the course 😊

All authorship is mentioned where possible.

# 📊 Exercises
## Exercise 1
The aim of the exercise is to get an overview of the basic capabilities of the Keras library and to build a simple neural network for MNIST dataset classification.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_01.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_01.ipynb)

## Exercise 2
The goal of the exercise is to learn how to solve regression problems using deep learning.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_02.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_02.ipynb)


## Exercise 3
The aim of the exercise is to learn how to use the basic architecture based on convolutional layers and how to classify image data.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_03.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_03.ipynb)


## Exercise 4
The aim of the exercise is to learn how to use transfer learning for image data, in the second part of the exercise we will look at time series classification using CNN.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_04.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_04.ipynb)

## Exercise 5
The goal of the exercise is to learn how to use Autoencoder and Variational autoencoder architectures in image data to generate new image data instances and detect anomalies.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_05.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_05.ipynb)

## Exercise 6
The goal of this exercise is to learn how to use recurrent neural networks for sentiment analysis of text data. In the exercise, we will work with data from Twitter.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_06.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_06.ipynb)

## Exercise 7
The goal of this exercise is to learn how to create your own Word2Vec embedding and generate your own text using RNN.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_07.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_07.ipynb)

## Exercise 8
The aim of the exercise is to learn how to work with the Attention mechanism within the RNN.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_08.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_08.ipynb)

## Exercise 9
The exercise focuses on the use of transformer models for text data classification using the Hugging face API.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_09.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_09.ipynb)

## Exercise 10
This lecture is focused on using CNN for object localization tasks.

> [Jupyter Notebook](https://github.com/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_10.ipynb)

> [Google Colab](https://colab.research.google.com/github/rasvob/VSB-FEI-Deep-Learning-Exercises/blob/main/dl_10.ipynb)

# 💡 Notes
## How to create a Python Virtual Enviroment named `venv`
### Create `venv`
```
python -m venv venv
```

### Activate `venv`

* Activate `venv` in **Windows**
```
.\venv\Scripts\Activate.ps1
```

* Activate `venv` in **Linux**
```
source venv/bin/activate
```


### Intall python packages

```
pip install jupyter "jupyterlab>=3" "ipywidgets>=7.6"
pip install pandas matplotlib requests seaborn scipy scikit-learn optuna plotly==5.18.0 tensorflow[and-cuda]
```

### Test TF2 installation

* It should print list of all your GPUs
    * It is not working if an empty list `[]` is printed

```python
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```