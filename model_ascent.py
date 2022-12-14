from keras import layers, regularizers
from keras.models import Sequential


import keras.backend as kb

def custom_loss(y_actual,y_pred):
    custom_loss=kb.square(y_actual-y_pred)
    return custom_loss*(-1)

def create_model_ascent(data):
    if data=="cifar10":
        model = Sequential()
        
        model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
        model.add(layers.GroupNormalization())
        model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization())
        model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization())
        model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.GroupNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax', kernel_regularizer = regularizers.l2(0.0005)))
        model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
        
        return model
    if data=="emnist":
        model = Sequential()
        
        model.add(layers.Conv2D(28, (3,3), padding='same', activation='relu', input_shape=(28,28,1)))
        model.add(layers.BatchNormalization(scale=False, center=False))
        model.add(layers.Conv2D(28, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(scale=False, center=False))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(56, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(scale=False, center=False))
        model.add(layers.Conv2D(56, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(scale=False, center=False))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Conv2D(112, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(scale=False, center=False))
        model.add(layers.Conv2D(112, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization(scale=False, center=False))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(112, activation='relu'))
        model.add(layers.BatchNormalization(scale=False, center=False))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(47, activation='softmax', kernel_regularizer = regularizers.l2(0.0005)))
        model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
        
        return model