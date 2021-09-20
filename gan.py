# Unzip dataset folder with images
!unzip monet_jpg.zip -d monet_jpg

'''
HANDLING IMAGES
'''

import os
import numpy as np
from PIL import Image

IMAGE_DIR = '/content/monet_jpg'

images_path = IMAGE_DIR
IMAGE_SIZE = 128 # rows/cols
IMAGE_CHANNELS = 3

training_data = []

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    training_data.append(np.asarray(image))

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1
np.save('artwork_data.npy', training_data)


from tensorflow import keras
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100
# Size vector to generate images from
NOISE_SIZE = 100
# Configuration
EPOCHS = 4000 # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3


'''
DISCRIMINATOR
'''
def build_discriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
    input_shape=image_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    #model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    #model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    #model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)

'''
GENERATOR
'''
def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation='relu', input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())
    #model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    #model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    for i in range(GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding='same'))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation('relu'))

    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding='same'))
    model.add(Activation('tanh'))
    input = Input(shape=(noise_size,))
    generated_image = model(input)
    
    return Model(input, generated_image)
    
    
 '''
IMAGE SAVER
'''
def save_images(cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c +
                        IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1

    output_path = 'content/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path, f"painting-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
    
    
'''
TRAINING MODEL
'''
image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
optimizer = Adam(0.0002, 0.5)

discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy',
optimizer=optimizer, metrics=['accuracy'])
generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)

discriminator.trainable = True

validity = discriminator(generated_image)
combined = Model(random_input, validity)
combined.compile(loss='binary_crossentropy',
optimizer=optimizer, metrics=['accuracy'])
y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))
fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
cnt = 1

gen_loss = []
disc_loss = []

for epoch in range(EPOCHS):
    idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
    x_real = training_data[idx]
    
    noise= np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
    x_fake = generator.predict(noise)
    
    discriminator_loss_real = discriminator.train_on_batch(x_real, y_real)

    discriminator_loss_generated = discriminator.train_on_batch(
    x_fake, y_fake)

    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)
    generator_loss = combined.train_on_batch(noise, y_real)

    disc_loss.append(discriminator_loss[1])
    gen_loss.append(generator_loss[1])

    if epoch > 1500:
      if epoch % SAVE_FREQ == 0:
          save_images(cnt, fixed_noise)
          cnt += 1
  
          print(f'{epoch} epoch, Discriminator loss: {100*  discriminator_loss[1]}, Generator loss: {100 * generator_loss[1]}')
      # check if finished
      # discriminator loss around 0.5 to 0.8
      # generator loss around 1.0 to 2.0
      # this is only valid if we have done some epochs and reached some form of stabilization
      if 0.5 <= discriminator_loss[1] <= 0.8 and 1.0 <= generator_loss[1] <= 2.0:

          print('EQUILIBRIUM REACHED. PROGRAM FINSISHED')

          save_images(cnt, fixed_noise)
          cnt += 1

          break


ep = [i for i in range(EPOCHS)]

plt.plot(ep, gen_loss, label='Generator loss')
plt.plot(ep, disc_loss, label='Discriminator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
