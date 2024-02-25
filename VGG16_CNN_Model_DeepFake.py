
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
import cv2
import numpy as np

data_dir = r".\DMML2\Celeb-DF"

img_height=224
img_width=224
num_channels=3

# Read celeb-df data from directories
train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='nearest')
                             


train_generator = train_datagen.flow_from_directory(
        data_dir+ '/TRAIN',
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='nearest')
                             
validation_generator = validation_datagen.flow_from_directory(
        data_dir+ '/TEST',
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')




#Load VGG16 model
input_tensor = Input(shape=(img_height, img_width, 3))
base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

# Create CNN model
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu',padding='valid'),
    MaxPooling2D((1, 1)),
    Conv2D(64, (3, 3), activation='relu',padding='valid'),
    MaxPooling2D((1, 1)),
    Conv2D(256, (3, 3), activation='relu',padding='valid'),
    MaxPooling2D((1, 1)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Combine models and create a master model
x = base_model.output
x = model(x)
combined_model = Model(inputs=base_model.input, outputs=x)


print(model.summary())
print(base_model.summary())

# Fit the model
# combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),tf.keras.metrics.AUC()])

history = combined_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

print(combined_model.summary())

#save the model
# combined_model.save("Combined_modelV1.h5")
# history.save("historyV1.h5")


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for AUC
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('model auc')
plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




# # Define the functions for noise addition

# Define the test data directory
test_dir=r".\DMML2\Celeb-DF"

# Define the noise function
def add_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_type == 'salt_and_pepper':
        row, col, ch = image.shape
#         s_vs_p = 0.5
#         amount = 0.004
        s_vs_p =0.001
        amount = 0.0001
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    else:
        raise ValueError('Noise type not supported')


# ## Applying and evaluating salt paper noise

import os
if not os.path.exists(test_dir+r'\noisy_SP\FAKE'):
    os.makedirs(test_dir+r'\noisy_SP\FAKE')
if not os.path.exists(test_dir+r'\noisy_SP\REAL'):
    os.makedirs(test_dir+r'\noisy_SP\REAL')
    
# Add noise to each image in the test FAKE directory

for file in os.listdir(test_dir+r'\TEST\FAKE'):
#     print(file)
    if file.endswith('.jpg'):
        image_path = os.path.join(test_dir+r'\TEST\FAKE', file)
        image = cv2.imread(image_path)
        noisy_image = add_noise(image, noise_type='salt_and_pepper')
        noisy_image_path = os.path.join(test_dir, r'noisy_SP\FAKE\\' + file)
        cv2.imwrite(noisy_image_path, noisy_image)
        
        
# Add noise to each image in the test REAL directory

for file in os.listdir(test_dir+r'\TEST\REAL'):
#     print(file)
    if file.endswith('.jpg'):
        image_path = os.path.join(test_dir+r'\TEST\REAL', file)
        image = cv2.imread(image_path)
        noisy_image = add_noise(image, noise_type='salt_and_pepper')
        noisy_image_path = os.path.join(test_dir, r'noisy_SP\REAL\\' + file)
        cv2.imwrite(noisy_image_path, noisy_image)



# Create the Noisy data generator
noisy_generator_sp = validation_datagen.flow_from_directory(data_dir + '/noisy_SP',
                                                   target_size=(img_height, img_width),
                                                   batch_size=32,
                                                   class_mode='binary')


print(combined_model.evaluate(noisy_generator_sp,steps=noisy_generator_sp.samples/noisy_generator_sp.batch_size))

noisy_SP_loss, noisy_SP_acc,noisy_SP_prec,noisy_SP_recall,noisy_SP_auc = combined_model.evaluate(noisy_generator_sp,
                                      steps=noisy_generator_sp.samples/noisy_generator_sp.batch_size)

print(noisy_SP_loss, noisy_SP_acc,noisy_SP_prec,noisy_SP_recall,noisy_SP_auc)


# ## Applying and evaluating gaussian noise


if not os.path.exists(test_dir+r'\noisy_gauss\FAKE'):
    os.makedirs(test_dir+r'\noisy_gauss\FAKE')
if not os.path.exists(test_dir+r'\noisy_gauss\REAL'):
    os.makedirs(test_dir+r'\noisy_gauss\REAL')
# Add noise to each image in the test set directory


for file in os.listdir(test_dir+r'\TEST\FAKE'):
#     print(file)
    if file.endswith('.jpg'):
        image_path = os.path.join(test_dir+r'\TEST\FAKE', file)
        image = cv2.imread(image_path)
        noisy_image = add_noise(image, noise_type='gaussian')
        noisy_image_path = os.path.join(test_dir, r'noisy_gauss\FAKE\\' + file)
        cv2.imwrite(noisy_image_path, noisy_image)
        
        
# Add noise to each image in the test set directory


for file in os.listdir(test_dir+r'\TEST\REAL'):
#     print(file)
    if file.endswith('.jpg'):
        image_path = os.path.join(test_dir+r'\TEST\REAL', file)
        image = cv2.imread(image_path)
        noisy_image = add_noise(image, noise_type='gaussian')
        noisy_image_path = os.path.join(test_dir, r'noisy_gauss\REAL\\' + file)
        cv2.imwrite(noisy_image_path, noisy_image)

# Create the Noisy data generator
noisy_generator_gauss = validation_datagen.flow_from_directory(data_dir + '/noisy_gauss',
                                                   target_size=(img_height, img_width),
                                                   batch_size=32,
                                                   class_mode='binary')

noisy_gauss_loss, noisy_gauss_acc,noisy_gauss_prec,noisy_gauss_recall,noisy_gauss_auc = combined_model.evaluate(noisy_generator_gauss,
                                      steps=noisy_generator_gauss.samples/noisy_generator_gauss.batch_size)

print(noisy_gauss_loss, noisy_gauss_acc,noisy_gauss_prec,noisy_gauss_recall,noisy_gauss_auc)


