'''
this is a faster multi-speaker LipNet Training script based off of the lipnet.py script
was taking forever to train before on multiple speakers (14+ hours) so we took steps to make it run faster (larger batch size, fewer epochs, and mixed precision (float16 to float32 only when helpful))
trained on speakers s2, s3, and s4 because s1 was off-center in the videos and we thought this might compromise performance on other speakers, the majority of whom were centered
'''

import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio

# mixed precision for faster training (only use float32 past certain point when helpful, otherwise use float16)
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

tf.config.list_physical_devices('GPU')

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


# 1. data loading

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret or frame is None:  # skip corrupted frames (we found many in s1)
            continue
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    # handle case where no valid frames were loaded or too few frames
    if len(frames) == 0:
        # return dummy frames filled with zeros
        frames = [tf.zeros((46, 140, 1), dtype=tf.float32) for _ in range(75)]
    elif len(frames) < 75:
        # pad with the last frame if we have some frames but not enough
        while len(frames) < 75:
            frames.append(frames[-1])
    elif len(frames) > 75:
        # trim to 75 frames if too many
        frames = frames[:75]

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    std = tf.maximum(std, 1e-8)
    return tf.cast((frames - mean), tf.float32) / std


# create vocab
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# translate back and forth between chars and numbs (based off of position in vocab)
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)


def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path: str):

    path = bytes.decode(path.numpy())

    # extract speaker and filename from path
    path_parts = path.split('/')
    speaker = path_parts[-2]
    file_name = path_parts[-1].split('.')[0]

    video_path = os.path.join('data', speaker, f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', speaker, f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments


def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


# 2. data pipeline

print("Loading data from speakers...")

data = tf.data.Dataset.list_files('./data/s*/*.mpg')
data = data.shuffle(2000, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(8, padded_shapes=([75, 46, 140, 1], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

# dataset size
total_samples = sum(1 for _ in data)
print(f"Total batches in dataset: {total_samples}")

# recreate dataset after counting
data = tf.data.Dataset.list_files('./data/s*/*.mpg')
data = data.shuffle(2000, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(8, padded_shapes=([75, 46, 140, 1], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

# split: train = 0.9, test = 0.1
train_size = int(total_samples * 0.9)
train = data.take(train_size)
test = data.skip(train_size)

print(f"Training batches: {train_size}")
print(f"Testing batches: {total_samples - train_size}")

# test pipeline
sample = data.as_numpy_iterator()
val = sample.next()

print(f"Video batch shape: {val[0].shape}")
print(f"Alignment batch shape: {val[1].shape}")

# create sample GIF
frames_to_save = val[0][0]
frames_uint8 = ((frames_to_save - frames_to_save.min()) / (frames_to_save.max() - frames_to_save.min()) * 255).astype(
    np.uint8)
frames_uint8 = np.squeeze(frames_uint8, axis=-1)
imageio.mimsave('./animation_fast.gif', frames_uint8, fps=10)


# 3. design NN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, \
    SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

# cast output to float32 for CTC loss (mixed precision issue)
model.add(
    Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax', dtype='float32'))

model.summary()


# 4. training

def scheduler(epoch, lr):
    
    if epoch < 20:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1)) # exponentially decrease LR from epoch 20 onwards


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75] * len(yhat), greedy=False)[0][0].numpy()
        for x in range(min(2, len(yhat))):  # show 2 examples
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8')) # original text
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8')) # predicted text for comparison
            print('~' * 100)


model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# load existing checkpoint to resume training
try:
    model.load_weights('models/checkpoint_fast.weights.h5')
    print("\n" + "=" * 100)
    print("Loaded existing checkpoint! Resuming training from where we left off...") # ended up needing this, training runs crashed overnight several times
    print("=" * 100 + "\n")
except:
    print("\n" + "=" * 100)
    print("No checkpoint found, starting training from scratch")
    print("=" * 100 + "\n")

checkpoint_callback = ModelCheckpoint(
    os.path.join('models', 'checkpoint_fast.weights.h5'),
    monitor='loss',
    save_weights_only=True
)

schedule_callback = LearningRateScheduler(scheduler)

example_callback = ProduceExample(test)

# fit model
model.fit(
    train,
    validation_data=test,
    epochs=50,
    callbacks=[checkpoint_callback, schedule_callback, example_callback]
)

print("\n" + "=" * 100)
print("Training complete!")
print("Model weights saved to: models/checkpoint_fast.weights.h5")
print("=" * 100 + "\n")


# 5. make predictions

model.load_weights('models/checkpoint_fast.weights.h5')

test_data = test.as_numpy_iterator()
sample = test_data.next()

yhat = model.predict(sample[0])

print('~' * 100, 'REAL TEXT')
print([tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in
       sample[1]])

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75] * len(yhat), greedy=True)[0][0].numpy()

print('~' * 100, 'PREDICTIONS')
print(
    [tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded])