import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
LOG_FILE = 'experiment_log.txt'
RESULTS_FILE = 'test_results.txt'


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    # print("Lables before:\n",labels)
    labels = tf.keras.utils.to_categorical(labels)
    # print(labels)
    # exit()
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    print("x_train, y_train", x_train.shape, y_train.shape)
    print("x_test, y_test", x_test.shape, y_test.shape)

    # Check if the log file exists
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        print("Log File found\n")
        with open(LOG_FILE, 'r') as log_file:
            # Read the last line to determine the last experiment completed
            last_experiment = log_file.readlines()[-1].strip()
            last_experiment_params = last_experiment.split(',')
            last_experiment_index = int(last_experiment_params[-1])
    else:
        last_experiment_index = -1  # Start from the beginning if the log file doesn't exist

    # Open log file for writing
    with open(LOG_FILE, 'a') as log_file, open(RESULTS_FILE, 'a') as results_file:
        # Loop over hyperparameters
        with open('experiment_parameters.txt', 'r') as params_file:
            for line in params_file:
                params = line.strip().split(';')
                num_conv_layers = int(params[0])
                filter_sizes = tuple(map(int, params[1].split(':')))
                pooling_layer = tuple(map(int, params[2].split(':')))
                num_hidden_layers = int(params[3])
                hidden_layer_size = tuple(map(int, params[4].split(':')))
                dropout_rate = float(params[5])

                # Skip if this experiment has already been completed
                if last_experiment_index >= 0 and last_experiment_index >= int(params[6]):
                    continue

                print(f"\nTraining model with hyperparameters:"
                      f"\nNum Conv Layers: {num_conv_layers}, Filter Sizes: {filter_sizes}, "
                      f"Pooling Layer: {pooling_layer}, "
                      f"Num Hidden Layers: {num_hidden_layers}, Hidden Layer Size: {hidden_layer_size}, "
                      f"Dropout Rate: {dropout_rate}")

                # Create and compile model based on hyperparameters
                model = get_model(num_conv_layers, filter_sizes, pooling_layer,
                                     num_hidden_layers, hidden_layer_size, dropout_rate)

                # Train the model
                model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

                # Evaluate the model
                results = model.evaluate(x_test, y_test)
                accuracy = results[1]

                # Write the results to the log file
                log_file.write(f"{num_conv_layers},{','.join(map(str, filter_sizes))},{pooling_layer},"
                               f"{num_hidden_layers},{hidden_layer_size},{dropout_rate},{last_experiment_index + 1}\n")

                # Write the results to the results file
                results_file.write(f"{num_conv_layers},{','.join(map(str, filter_sizes))},{pooling_layer},"
                                   f"{num_hidden_layers},{hidden_layer_size},{dropout_rate},{accuracy}\n")
                last_experiment_index = last_experiment_index+1
                print(f"Test Accuracy: {accuracy:.4f}")
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            images.append(img)
            labels.append(category)

    return np.array(images), np.array(labels)


def get_model(num_conv_layers, filter_sizes, pooling_layer, num_hidden_layers, hidden_layer_size, dropout_rate):
    model = tf.keras.Sequential()

    # Convolutional Layers
    for i in range(num_conv_layers):
        model.add(Conv2D(filters=filter_sizes[i], kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pooling_layer[i], pooling_layer[i])))

    model.add(Flatten())

    # Dense Hidden Layers
    for h in range(num_hidden_layers):
        model.add(Dense(hidden_layer_size[h], activation='relu'))
        model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
