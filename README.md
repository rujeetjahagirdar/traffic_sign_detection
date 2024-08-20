# Traffic Sign Classification with Neural Networks

## Project Overview
This project involves developing a neural network to classify traffic signs using TensorFlow. The goal is to accurately identify different types of traffic signs from images to support self-driving car technologies. The dataset used for this project is the German Traffic Sign Recognition Benchmark (GTSRB), which includes thousands of images of 43 different road signs.

## Dataset
The dataset is available at [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) and consists of images categorized into 43 directories, each representing a different road sign category. The images vary in size and are preprocessed to be 512x512 pixels.

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/rujeetjahagirdar/traffic_sign_detection.git
    cd traffic-sign-classification
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Download the data**:
    - Download the dataset from the provided link and place it in the `gtsrb` directory inside the project directory.

2. **Run the script**:
    ```bash
    python traffic.py gtsrb
    ```

    This command will train the model on the GTSRB dataset and evaluate its performance.

## Implementation Details
1. **Data Loading**:
    The `load_data` function reads and preprocesses images from the dataset, resizing them to a standard size and splitting them into training and testing sets.

2. **Model Building**:
    The `get_model` function constructs a neural network using TensorFlow, including layers such as convolutional, pooling, and fully connected layers. The model is trained to classify road signs into 43 categories.

3. **Experiments**:
    - Various architectures and hyperparameters were tested, including different numbers of convolutional layers, filter sizes, and dropout rates.

## Experimentation Details

# Configuration Variation: 
Various configurations of convolutional and pooling layers, filter sizes, and hidden layers were tested to identify the optimal architecture for the CNN.

# Observations:
Experimentation revealed that increasing the number of hidden layers and adjusting dropout rates played a crucial role in improving model accuracy.
Filter sizes and pooling layer variations were observed to have different effects on feature extraction and model robustness.

# Resume Functionality: 
The project features a resume function, allowing the continuation of experiments from the last completed iteration. This helps save time and resources during experimentation.

## Experimentation Steps

1. **Data Loading and Preprocessing:**
   - In this project we have worked on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which consists of different images of the traffic signs categorised into 43 categories.

2. **Model Architecture:**
   - For this project we have used CNN model. We have experimented with various architecture of the CNN network.
   - Convolutional Layers: Extracts hierarchical features from input images using learnable filters, capturing spatial patterns crucial for image understanding.

    Pooling Layers: Downsamples feature maps, reducing dimensionality and preserving essential information while enhancing computational efficiency.

    Hidden Layers: Enables the network to learn complex hierarchical representations, facilitating the abstraction of high-level features for accurate classification.

3. **Hyperparameter Tuning:**
   - In this project we have exprimented with various hyperparameters such as filter sizes, pool sizes, and dropout rates.

4. **Training and Evaluation:**
   - We have used 'ADAM' optimizer along with categorical_crossentropy loss function. 

5. **Experiment Results:**
   - We have observed that, the dropout rate of the model is affecting the result significantly. As we increased the dropout rate, the model seems to not perform well.

## What Worked Well

- Choosing different hyperparameters like 3 convolution layers of various size (32, 64, 128), and dropout rate of 0.4 seems to work well.
- This was the result of one of the experiment that we have performed.

## What Didn't Work Well

- We have observed that increasing the dropout rate reduceses the performance of the model.

## Input and Output

- Experiment Input File: The project reads experiment parameters from an input file (experiment_parameters.txt) in the format specified.

- Results Output File: The results of each experiment, including accuracy, are written to a log file (experiment_log.txt) for future reference.


## Report
A detailed report on the experimentation process, model performance, and results is included. The report discusses various approaches, what worked well, and areas for improvement.

## Contributing
Feel free to fork the repository and submit pull requests with improvements or additional features.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## References
1. [TensorFlow Keras Overview](https://www.tensorflow.org/tutorials/images/cnn)
2. [OpenCV-Python Documentation](https://opencv.org/)
3. [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/whenamancodes/wild-animals-images)
