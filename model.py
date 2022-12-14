from keras import layers, regularizers
from keras.models import Sequential


def create_model(data):
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
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    if data=="emnist" or data =="femnist":
        model = Sequential()
    
        model.add(layers.Conv2D(28, (3,3), padding='same', activation='relu', input_shape=(28,28, 1)))
        model.add(layers.GroupNormalization(28))
        model.add(layers.Conv2D(28, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization(28))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(56, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization(28))
        model.add(layers.Conv2D(56, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization(28))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Conv2D(112, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization(28))
        model.add(layers.Conv2D(112, (3,3), padding='same', activation='relu'))
        model.add(layers.GroupNormalization(28))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(112, activation='relu'))
        model.add(layers.GroupNormalization(28))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(62, activation='softmax', kernel_regularizer = regularizers.l2(0.0005)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

