import keras
import keras_cv
import keras_tuner
import kecam
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.models import Model
from keras import layers, activations
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import plot_model
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras.models import Model
from keras_cv_attention_models import eva02
import matplotlib.pyplot as plt

(ds_test, ds_val, ds_train), ds_info = tfds.load('stanford_dogs', split= ["test[0%:50%]", "test[50%:]", "train"], shuffle_files=True, data_dir='tensorflow_datasets/', as_supervised=True, with_info=True)

image_size = (336, 336)
batch_size = 32

rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 1), augmentations_per_image=3, magnitude=0.5, rate=0.9090909090909091)

def normalize_img(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, image_size, method='nearest')
  image = image / 255.
  label = tf.one_hot(label, 120)
  return image, label


def augment(image, label):
  image = rand_augment(image)
  return image, label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.map(
    augment, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.cache()
ds_val = ds_val.batch(batch_size)
ds_val= ds_val.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model('EVA02Small_warmedup.keras')

layers = model.layers
for layer in layers:
  if 'block11_mlp_dense_2' in layer.name:
    layer.trainable = True

#for layer in layers:
#  if 'block11_mlp_dense_2' in layer.name 'block11_mlp_dense_gate' in layer.name:
#    layer.trainable = True

#for layer in layers:
#  if 'block11' in layer.name:
#    layer.trainable = True

#for layer in layers:
#  if 'block11' in layer.name or 'block10_mlp_dense_gate' in layer.name:
#    layer.trainable = True

checkpoint_filepath='checkpoint.weights.h5'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.002, restore_best_weights=False)

model.compile(loss='categorical_crossentropy',
            optimizer= Adam(learning_rate=0.00005),
            metrics= ['accuracy'])

history=model.fit(
        ds_train,
        epochs=200,
        validation_data=ds_val,
        callbacks=[stop_early, model_checkpoint_callback]
        )
