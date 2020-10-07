import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

"""
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=4,
  batch_size=32,
)

model.evaluate(
  test_images,
  to_categorical(test_labels)
)

# Save weights
model.save_weights('model.h5')
"""


# Load saved weights
model.load_weights('model.h5')


# Load and prep image
def load_image(filename):
	img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
	img = img_to_array(img)
	img = img.reshape((-1, 784))
	img = img.astype('float32')
	img = (img / 255) - 0.5
	return img

example_test = load_image('Examples/Eight.png')


# predicts
predictions = model.predict(example_test)

# Print models predictions
print(np.argmax(predictions, axis=1))