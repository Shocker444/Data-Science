import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Input, Reshape, Lambda, Layer, Flatten
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve,auc
from sklearn.metrics import confusion_matrix
import seaborn as sns


mysize = (28, 28)
sub = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

def read_images_from_folder(folder_path):
    image_list = []
    label_list = []

    subfolders = os.listdir(folder_path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            images = os.listdir(subfolder_path)
            for image_file in images:
                image_path = os.path.join(subfolder_path, image_file)
                if os.path.isfile(image_path):
                    image = Image.open(image_path).convert('L')
                    image = image.resize(mysize)
                    image = np.array(image)
                    image = image.astype('float32')
                    image /= 255
                    if image is not None:
                        image_list.append(image)
                        label_list.append(subfolder)

    return image_list, label_list

#folder_path = 'C:\\Users\\HP\\Desktop\\datasets\\CRC-VAL-HE-7K'

#train_folder_path = os.path.join(folder_path, 'train')
#train_images, train_labels = read_images_from_folder(train_folder_path)

#test_folder_path = os.path.join(folder_path, 'test')
#test_images, test_labels = read_images_from_folder(test_folder_path)

#trainX = np.array(train_images).reshape(-1, 28, 28, 1)
#trainY = np.array(train_labels)

#testX = np.array(test_images).reshape(-1, 28, 28, 1)
#testY = np.array(test_labels)

#label_encoder = LabelEncoder()

#trainY_encoded = label_encoder.fit_transform(trainY)
#testY_encoded = label_encoder.transform(testY)

#trainY_categorical = to_categorical(trainY_encoded, num_classes=9)
#testY_categorical = to_categorical(testY_encoded, num_classes=9)

test = tf.random.normal([1, 28, 28, 1])

inputs = Input(shape=(28, 28, 1))

conv1 = Conv2D(256, (9, 9), activation='relu', padding='valid')(inputs)
conv2 = Conv2D(256, (9, 9), strides=2, padding='valid')(conv1)
reshaped = Reshape((6 * 6 * 32, 8))(conv2)

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

squashed_output = Lambda(squash, output_shape=(6 * 6 * 32, 8))(reshaped)

class DigitCapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(DigitCapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get('glorot_uniform')

    def build(self, input_shape):
        self.W = self.add_weight(shape=[1, input_shape[1], self.num_capsule, self.dim_capsule, input_shape[2]],
                                 initializer=self.kernel_initializer,
                                 name='W')
        
        self.built=True
       

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, -2)
        inputs_expand = tf.expand_dims(inputs_expand, -1)
        inputs_hat = tf.matmul(self.W, inputs_expand)
        #inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        #inputs_hat = tf.map_fn(lambda x: tf.linalg.matvec(x, self.W, [2, 3]), elems=inputs_tiled)
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], K.shape(inputs_hat)[1], self.num_capsule])
        print(inputs_hat.shape)

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=-2)
            outputs = squash(tf.reduce_sum(c[:, :, :, tf.newaxis] * tf.squeeze(inputs_hat, axis=-1), axis=1, keepdims=True))
            if i < self.routings - 1:
                b += tf.squeeze(tf.matmul(tf.expand_dims(inputs_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4])

        return K.reshape(outputs, [-1, self.num_capsule, self.dim_capsule])

digit_caps = DigitCapsuleLayer(num_capsule=10, dim_capsule=16, routings=3)(squashed_output)
outputs = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1) + K.epsilon()))(digit_caps)

'''def mask(outputs):
    if type(outputs) != list:
        norm_outputs = K.sqrt(K.sum(K.square(outputs), axis=-1) + K.epsilon())
        y = K.one_hot(indices=K.argmax(norm_outputs, axis=1), num_classes=10)
        y = Reshape((10, 1))(y)
        return Flatten()(y * outputs)
    else:
        y = Reshape((10, 1))(outputs[1])
        masked_output = y * outputs[0]
        return Flatten()(masked_output)

masked = Lambda(mask)([digit_caps, inputs])
masked_for_test = Lambda(mask)(digit_caps)

decoder_inputs = Input(shape=(16 * 10,))
dense1 = Dense(512, activation='relu')(decoder_inputs)
dense2 = Dense(1024, activation='relu')(dense1)
decoded_outputs = Dense(784, activation='sigmoid')(dense2)
decoded_outputs = Reshape((28, 28, 1))(decoded_outputs)

def loss_fn(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, axis=1))

train_decoder_inputs = np.zeros((trainX.shape[0], 16 * 10))
test_decoder_inputs = np.zeros((testX.shape[0], 16 * 10))

capsule_model = models.Model([inputs, decoder_inputs], [outputs, decoded_outputs])
capsule_model.compile(optimizer='adam', loss=[loss_fn, 'mse'], loss_weights=[1., 0.0005], metrics=['accuracy'])

capsule_model.fit([trainX, train_decoder_inputs], [trainY_categorical, trainX],
                  batch_size=128, epochs=5, validation_data=([testX, test_decoder_inputs], [testY_categorical, testX]))'''
