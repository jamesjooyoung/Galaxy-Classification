from types import SimpleNamespace
import numpy as np
import tensorflow as tf

def get_single_CNN_model():
    Conv2D = tf.keras.layers.Conv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Dropout = tf.keras.layers.Dropout
    
    input_prep_fn = tf.keras.Sequential(
        [
            # add center crop
            tf.keras.layers.CenterCrop(224, 224),
            # add resizing
            tf.keras.layers.Resizing(69, 69),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=10, output_mode="one_hot"
    )

    augment_fn = tf.keras.Sequential(
        [            
         # Random rotation(45)
         tf.keras.layers.RandomRotation(45.0),
         # Color jitter (0, 0.2)
         # Random horizontal and vertical flip
         tf.keras.layers.RandomFlip("horizontal_and_vertical"),
         # Normalize
        ]
    )

    model = CustomSequential(
        [
        # conv layer with 16 filters
        Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        # dropout layer
        Dropout(rate = 0.1),
        # flatten layer
        tf.keras.layers.Flatten(),
        # dropout layer
        Dropout(rate = 0.1),
        # dense layer
        tf.keras.layers.Dense(100, activation='leaky_relu'),
        # dropout layer
        Dropout(rate=0.1),
        # final dense layer using softmax
        tf.keras.layers.Dense(10, activation='softmax')
        ], input_prep_fn=input_prep_fn, output_prep_fn=output_prep_fn, augment_fn=augment_fn
    )


    model.compile(
        # using Adam as the optimizer
        optimizer="adam", # maybe change learning rate to 0.0001
        # loss is categorical cross-entropy
        loss="categorical_crossentropy", 
        # using accuracy as the metric
        metrics=["categorical_accuracy"],
    )

    return SimpleNamespace(model=model, epochs=50, batch_size=100)


def get_multi_CNN_model():

    Conv2D = tf.keras.layers.Conv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Dropout = tf.keras.layers.Dropout
    
    input_prep_fn = tf.keras.Sequential(
        [
            # add center crop
            tf.keras.layers.CenterCrop(224, 224),
            # add resizing
            tf.keras.layers.Resizing(69, 69),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        # one hot encode labels
        num_tokens=10, output_mode="one_hot"
    )

    augment_fn = tf.keras.Sequential(
        [            
         # Random rotation(45)
         tf.keras.layers.RandomRotation(45.0),
         # Color jitter (0, 0.2)
         # Random horizontal and vertical flip
         tf.keras.layers.RandomFlip("horizontal_and_vertical"),
         # Normalize
        ]
    )

    model = CustomSequential(
        [
        # conv layer with 16 filters
        Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # conv layer with 64 filters
        Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # conv layer with 128 filters
        Conv2D(filters=128, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        # dropout layer
        Dropout(rate = 0.1),
        # flatten layer
        tf.keras.layers.Flatten(),
        # dropout layer
        Dropout(rate = 0.1),
        # dense layer
        tf.keras.layers.Dense(100, activation='leaky_relu'),
        # dropout layer
        Dropout(rate=0.1),
        # final dense layer using softmax
        tf.keras.layers.Dense(10, activation='softmax')
        ], input_prep_fn=input_prep_fn, output_prep_fn=output_prep_fn, augment_fn=augment_fn
    )


    model.compile(
        # using Adam as the optimizer
        optimizer="adam", # maybe change learning rate to 0.0001
        # loss is categorical cross-entropy
        loss="categorical_crossentropy", 
        # using accuracy as the metric
        metrics=["categorical_accuracy"],
    )

    return SimpleNamespace(model=model, epochs=1, batch_size=100)


# using a custom class CustomSequential in order to modify input and output data, while adding in data augmentation
class CustomSequential(tf.keras.Sequential):
# subclass of the tf.keras.Sequential model

    def __init__(
        self,
        *args,
        input_prep_fn=lambda x: x,
        output_prep_fn=lambda x: x,
        augment_fn=lambda x: x,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # takes in input_prep_fn, output_prep_fn, augment_fn
        self.input_prep_fn = input_prep_fn
        self.output_prep_fn = output_prep_fn
        self.augment_fn = augment_fn

    # define the batch step
    def batch_step(self, data, training=False):
        # take in the data
        x_raw, y_raw = data
        # preprocess the input
        x = self.input_prep_fn(x_raw)
        # one_hot encode the output
        y = self.output_prep_fn(y_raw)

        # add in augmentation only for training
        if training:
            x = self.augment_fn(x)

        # inside gradient tape
        with tf.GradientTape() as tape:
            # calculate y_pred
            y_pred = self(x, training=training)
            # calculate the compiled loss (in this case, categorical cross-entropy)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if training:
            # Compute the gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            # apply gradients to the optimizer
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update the state of the metrics
        self.compiled_metrics.update_state(y, y_pred)
        # return the metrics and their values at each epoch
        return {m.name: m.result() for m in self.metrics}

    # train step function
    def train_step(self, data):
        return self.batch_step(data, training=True)

    # test step function
    def test_step(self, data):
        return self.batch_step(data, training=False)

    # predict step function
    def predict_step(self, inputs):
        x = self.input_prep_fn(inputs)
        return self(x)