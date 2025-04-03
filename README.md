# Handwritten Mathematical Equation Recognition

This project is aimed at recognizing handwritten mathematical equations and solving them using deep learning techniques. The system processes images of handwritten equations, breaks them down into individual mathematical symbols (digits, operators, etc.), and reconstructs the equations using a trained model.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)


## Background

This project uses **deep learning models** for handwritten mathematical equation recognition. The core of the project is built on **Convolutional Neural Networks (CNNs)** for feature extraction and **Long Short-Term Memory (LSTM)** networks for sequence prediction.

Key features of the project include:
- **Equation Recognition**: Identifies handwritten digits, operators, and variables.
- **Deep Learning**: The model is trained on a dataset of mathematical symbols and equations.
- **Web Interface**: Allows users to upload images of handwritten equations and view predictions.

## Install

To set up the project, you'll need Python and some dependencies. To install them, run:

```sh
pip install -r requirements.txt
The dependencies include libraries for image processing, machine learning, and the web interface.

Usage
Clone the repository:

sh
Copy
Edit
git clone https://github.com/yourusername/handwritten-math-recognition.git
cd handwritten-math-recognition
Install the required dependencies:

sh
Copy
Edit
pip install -r requirements.txt
To run the project, navigate to the Front_End directory, and open index.html in your web browser. You can also train the model by running the Jupyter notebooks in Stage 01 to Stage 05.

For inference, upload an image containing a handwritten equation, and the system will output the predicted equation.

Features
Handwritten Equation Recognition: The system processes and recognizes handwritten equations.

Deep Learning-Based Model: The model combines CNNs and LSTMs for better recognition and sequence prediction.

Web Interface: Users can upload images of equations and see predictions interactively.

Equation Reconstruction: Recognized symbols are put together to form the original equation.

Technologies Used
Machine Learning: TensorFlow, Keras

Image Processing: OpenCV

Web Interface: HTML, CSS, JavaScript

Model: CNN + LSTM architecture for recognition and sequence prediction

Project Structure
plaintext
Copy
Edit
fyp/
│
├── Front_End/                     # Web interface to interact with the model
│   ├── index.html                 # Main HTML page for the frontend
│   ├── script.js                  # JavaScript for frontend logic
│   ├── style.css                  # Styling for the frontend
│   ├── model.json                 # Model metadata for inference
│   ├── model.py                   # Backend for loading the model
│   ├── group1-shard1of2.bin       # Model weights (part 1)
│   └── group1-shard2of2.bin       # Model weights (part 2)
│
├── Stage 01/                      # Initial implementation and data prep
│   ├── DataPrep.ipynb             # Dataset preparation notebook
│   └── training_model.ipynb       # Initial model training
│
├── Stage 02/                      # Improved model training
│   ├── training_model_04.ipynb    # Model training and improvements
│
├── Stage 03/                      # Saved model and dataset updates
│   ├── Dataset.txt                # Final dataset information
│   ├── data_preparation_03.ipynb  # Final data preparation
│   ├── model_hand.h5              # Saved model
│   └── training_model_04.ipynb    # Model training notebook
│
├── Stage 04/                      # Additional model tuning and image processing
│   ├── Changes.txt                # Parameter changes
│   ├── training_model_04.ipynb    # Model and image processing visualization
│   ├── training_model_05.ipynb    # Trial for BLSTM
│   └── training_model_06.ipynb    # Additional training trial
│
├── Stage 05/                      # Final improvements and best performing model
│   ├── data_preparation_03.ipynb  # Latest dataset preparation
│   ├── model_lstm.h5              # Best performing model
│   └── training_model_07.ipynb    # Final training model
│
└── Survey Paper.pdf               # Updated survey paper
