'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# Define the Model
mymodel=Sequential()
mymodel.add(Conv2D(32,(3,3),activation='relu', input_shape=(64, 64, 1)))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Flatten())
mymodel.add(Dense(1,activation='sigmoid'))
mymodel.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
#Organize the data
train=ImageDataGenerator(rescale=1./255)
test=ImageDataGenerator(rescale=1./255)
train_set=train.flow_from_directory('train',target_size=(64,64),batch_size=32,class_mode='binary')
test_set=train.flow_from_directory('test',target_size=(64,64),batch_size=32,class_mode='binary')
#Train the modal
mymodel.fit(train_set,epochs=10,validation_data=test_set)
#Save the model
mymodel.save('eye.h5',mymodel)'''



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),  # Change input shape to (64, 64, 1)
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Organize the data
train=ImageDataGenerator(rescale=1./255)
test=ImageDataGenerator(rescale=1./255)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have training and validation directories with drowsy and non-drowsy subdirectories
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(64, 64),
    color_mode='grayscale',  # Keep color_mode='grayscale'
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'test',
    target_size=(64, 64),
    color_mode='grayscale',  # Keep color_mode='grayscale'
    batch_size=32,
    class_mode='binary'
)

k = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
model.save('eye.h5')
