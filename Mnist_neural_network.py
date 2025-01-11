import tensorflow as tf
import tensorflow_datasets as tfds

# Load the MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',  # Name of the dataset
    split=['train', 'test'],  # Split into training and testing sets
    shuffle_files=True,  # Shuffle files for better training performance
    as_supervised=True,  # Return data as (image, label) tuples
    with_info=True,  # Load additional metadata about the dataset
)
##########################################################################

"""
    Extract dataset metadata:
        Number of classes (digits 0-9)
        Image shape (28x28 grayscale images)
        Number of training examples
        Number of testing examples
"""
num_of_classes = ds_info.features['label'].num_classes
image_shape = ds_info.features['image'].shape
train_size = ds_info.splits['train'].num_examples
test_size = ds_info.splits['test'].num_examples
################################################################################

# Print dataset metadata
print(num_of_classes)
print(image_shape)
print(train_size)
print(test_size)
################################################################################

# Function for normalizing images
def normalize_img(image, label):
    """
    Normalize images by converting `uint8` type to `float32`
    and scaling pixel values to the range [0, 1].
    """
    return tf.cast(image, tf.float32) / 255., label
##########################################################################

"""
    Preprocess the training dataset
        Normalize images
        Cache data in memory for faster training
        Shuffle the dataset
        Create batches of size 32
        Prefetch data for better performance
"""
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(48)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
############################################################################

"""
    Preprocess the testing dataset
        Normalize images
        Create batches of size 128
        Cache data in memory
        Prefetch data for faster testing
"""
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
##############################################################################
# Define network architecture with a deeper structure and regularization
layers = [
    tf.keras.layers.Flatten(input_shape=image_shape),  # Flatten the input (28x28 images -> 784 vector)
    tf.keras.layers.Dense(94, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),  # Dense layer with L2 regularization
    tf.keras.layers.Activation('relu'),  # Leaky ReLU activation for non-linearity
    tf.keras.layers.Dropout(0.1),  # Dropout to prevent overfitting
    ##################################################################################
    tf.keras.layers.Dense(49, kernel_regularizer=tf.keras.regularizers.l2(0.0004)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    ##################################################################################
    tf.keras.layers.Dense(81, kernel_regularizer=tf.keras.regularizers.l2(0.008)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.25),
    ##################################################################################
    tf.keras.layers.Dense(33, kernel_regularizer=tf.keras.regularizers.l2(0.003)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.1),
    ##################################################################################
    tf.keras.layers.Dense(num_of_classes),  # Output layer with one node per class
    tf.keras.layers.Softmax()  # Softmax's activation for probabilistic class outputs
]

# Create a sequential model using the defined layers
model = tf.keras.models.Sequential(layers)

# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Loss function for multi-class classification
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  # Metric for evaluation: accuracy
)

# Display the model summary
model.summary()

# Train the model
model.fit(
    ds_train,  # Training dataset
    epochs=65,  # Number of epochs to train
    validation_data=ds_test  # Validation dataset
)
