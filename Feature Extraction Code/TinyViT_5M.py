import keras
import keras_cv
import kecam
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.models import Model
from keras import layers, activations
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras.models import Model
from keras_cv_attention_models import tinyvit
import matplotlib.pyplot as plt

(ds_test, ds_val, ds_train), ds_info = tfds.load('stanford_dogs', split= ["test[0%:50%]", "test[50%:]", "train"], shuffle_files=False, data_dir='tensorflow_datasets/', as_supervised=True, with_info=True)

image_size = (224, 224)
batch_size = 32

def normalize_img(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, image_size, method='nearest')
  image = image / 255.
  label = tf.one_hot(label, 120)
  return image, label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
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

base_model = tinyvit.TinyViT_5M(num_classes=0, pretrained="imagenet")
base_model.trainable = False
x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)

predictions = Dense(120, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy',
            optimizer= Adam(learning_rate=0.0001),
            metrics= ['accuracy'])

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.005)

history=model.fit(
        ds_train,
        epochs=100,
        validation_data=ds_val,
        callbacks=[stop_early]
        )
