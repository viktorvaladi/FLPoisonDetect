import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import copy
import keras.backend as kb

class FedProx:
    def __init__(self, pga):
        self.model = self.get_model()
        self.optimizer = keras.optimizers.Adam()
        if pga:
            self.loss_fn = self.custom_loss
        else:
            self.loss_fn = keras.losses.categorical_crossentropy
        self.batch_size = 125
        self.epochs = 10
    
    def custom_loss(self, y_actual,y_pred):
        custom_loss=kb.square(y_actual-y_pred)
        return custom_loss*(-1)
    
    def fit(self, parameters, x_train, y_train, newold):
        initial_weights = copy.deepcopy(parameters)
        val_split = int(len(x_train)*0.1)
        x_val = x_train[-val_split:]
        y_val = y_train[-val_split:]
        x_train = x_train[:-val_split]
        y_train = y_train[:-val_split]
        self.model.set_weights(parameters)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size)

        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = self.model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = self.loss_fn(y_batch_train, logits)
                    if newold == "fedprox":
                        mu = tf.constant(0.1, dtype=tf.float32)
                        prox_term =(mu/2)*self.difference_model_norm_2_square(self.model.get_weights(), initial_weights)
                        loss = loss_value + prox_term
                    else:
                        loss = loss_value

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return self.model.get_weights()
    
    def difference_model_norm_2_square(self, global_model, local_model):
        """Calculates the squared l2 norm of a model difference (i.e.
        local_model - global_model)
        Args:
            global_model: the model broadcast by the server
            local_model: the current, in-training model

        Returns: the squared norm

        """
        model_difference = tf.nest.map_structure(lambda a, b: a - b,
                                            local_model,
                                            global_model)
        squared_norm = tf.square(tf.linalg.global_norm(model_difference))
        return squared_norm


    def get_model(self):
        inputs = keras.Input(shape=(32,32,3), name="digits")
        x1 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
        x2 = layers.GroupNormalization()(x1)
        x3 = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x2)
        x4 = layers.GroupNormalization()(x3)
        x5 = layers.MaxPooling2D(pool_size=(2,2))(x4)
        x6 = layers.Dropout(0.3)(x5)
        x7 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x6)
        x8 = layers.GroupNormalization()(x7)
        x9 = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x8)
        x10 = layers.GroupNormalization()(x9)
        x11 = layers.MaxPooling2D(pool_size=(2,2))(x10)
        x12 = layers.Dropout(0.5)(x11)
        x13 = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x12)
        x14 = layers.GroupNormalization()(x13)
        x15 = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x14)
        x16 = layers.GroupNormalization()(x15)
        x17 = layers.MaxPooling2D(pool_size=(2,2))(x16)
        x18 = layers.Dropout(0.5)(x17)
        x19 = layers.Flatten()(x18)
        x20 = layers.Dense(128, activation='relu')(x19)
        x21 = layers.GroupNormalization()(x20)
        x22 = layers.Dropout(0.5)(x21)
        outputs = layers.Dense(10, activation='softmax', kernel_regularizer = regularizers.l2(0.0005))(x22)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
