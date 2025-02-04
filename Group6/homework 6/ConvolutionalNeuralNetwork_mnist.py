"""
Python Version: 3.8.5
Tensorflow version: 2.4.1
Keras is integrated into Tensorflow currently.
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import time
from tensorflow.keras.layers import SimpleRNN, Activation, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model



def trainning_process(model_history):
    fig = plt.figure(figsize=(16, 5))
    # Accuracy Increasing
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), model_history.history['accuracy'], 'blue')
    plt.plot(range(1, epochs + 1), model_history.history['val_accuracy'], 'red')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.xticks(np.arange(1, epochs + 1), fontsize=15)
    # Loss Decreasing
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), model_history.history['loss'], 'blue')
    plt.plot(range(1, epochs + 1), model_history.history['val_loss'], 'red')
    plt.title('Model Loss')
    plt.ylabel('Categorical Cross-Entropy', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.xticks(np.arange(1, epochs + 1), fontsize=15)
    plt.tight_layout()
    plt.show()
    return fig




# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
(X_train, y_train), (X_test, y_test) = mnist.load_data()

train_sample_size, row_size, col_size = X_train.shape
test_sample_size = X_test.shape[0]

print(f"Total Sample Size: {train_sample_size + test_sample_size}, "
      f"Training Sample Size: {train_sample_size},"
      f"Testing Sample Size: {test_sample_size}")
print(f"row pixel: {row_size}, column pixel: {col_size}")

# Parameter Specification


np.random.seed(123)  # for reproducibility
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape = (row_size, col_size, 1)
num_classes = 10
batch_size = 128  # Number of sample used to update gradient
epochs = 24  # Number of iterations

# data pre-processing
X_train = X_train.reshape(train_sample_size, *input_shape).astype('float32')
X_test = X_test.reshape(test_sample_size, *input_shape).astype('float32')
# Normalize pixel data
X_train = X_train / 255
X_test = X_test / 255
# Transform label set into binary representation, or so called "One-hot encoding"
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# build RNN model
model = Sequential()

# RNN model specification
model.add(Convolution2D(nb_filters, *kernel_size, padding='valid', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Convolution2D(nb_filters, *kernel_size, padding='valid'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=pool_size))  # Pool layer
model.add(Dropout(0.25))  # Randomly deactivate neurons
model.add(Flatten())  # Transform into 1 dimensional data
model.add(Dense(128))  # Full connected
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation("softmax"))

# Compile with defined parameters
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
start_time = time.time()
model_result = model.fit(X_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(X_test, y_test))
end_time = time.time()
print(f'Training takes {round((end_time - start_time) / 60, 1)} minutes to complete')
process_plot = trainning_process(model_history=model_result)
process_plot.savefig('trainning_process_plot.png', dpi=300)
validation_acc = model_result.history['val_accuracy'][-1]
print(f'The final accuracy is {validation_acc * 100}%')
# The final cross validation accuracy is 94.24%
model.save('DEDA_MachineLearning/cnn_model.h5') ##contains datasets, array like collections of data

# ============Load Trained Model=============
model_loaded = load_model('DEDA_MachineLearning/cnn_model.h5')
test_accu = model_loaded.evaluate(X_test, y_test)
print(f'Test Accuracy is: {test_accu[1]}')
