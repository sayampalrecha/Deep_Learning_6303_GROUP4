'''
Test script for LipNet model
Calculates Character Error Rate (CER) and Word Error Rate (WER)
'''

import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List, Tuple
import jiwer

# mixed precision for consistency with training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


# ============================================================================
# 1. Data Loading Functions (same as training)
# ============================================================================

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    if len(frames) == 0:
        frames = [tf.zeros((46, 140, 1), dtype=tf.float32) for _ in range(75)]
    elif len(frames) < 75:
        while len(frames) < 75:
            frames.append(frames[-1])
    elif len(frames) > 75:
        frames = frames[:75]

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    std = tf.maximum(std, 1e-8)
    return tf.cast((frames - mean), tf.float32) / std


# Create vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
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

    path_parts = path.split('/')
    speaker = path_parts[-2]
    file_name = path_parts[-1].split('.')[0]

    video_path = os.path.join('data', speaker, f'{file_name}.mpg')
    alignment_path = os.path.join('data','s5','alignments', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments


def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


# ============================================================================
# 2. Build Model (same architecture as training)
# ============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, \
    SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def build_model():
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

    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax', dtype='float32'))

    return model


# ============================================================================
# 3. Error Rate Calculation Functions
# ============================================================================

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate"""
    return jiwer.cer(reference, hypothesis)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    return jiwer.wer(reference, hypothesis)


def decode_predictions(yhat, input_length=75):
    """Decode CTC predictions to text"""
    decoded = tf.keras.backend.ctc_decode(
        yhat,
        input_length=[input_length] * len(yhat),
        greedy=True
    )[0][0].numpy()
    return decoded


def tokens_to_text(tokens) -> str:
    """Convert token arrays to text strings"""
    return tf.strings.reduce_join(num_to_char(tokens)).numpy().decode('utf-8')


# ============================================================================
# 4. Main Testing Function
# ============================================================================

def evaluate_model(model, test_dataset, num_batches=None):
    """
    Evaluate model on test dataset and calculate CER and WER

    Args:
        model: Trained LipNet model
        test_dataset: TensorFlow dataset with test data
        num_batches: Number of batches to evaluate (None = all batches)

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 100)
    print("Starting Model Evaluation")
    print("=" * 100 + "\n")

    all_cer = []
    all_wer = []
    all_references = []
    all_hypotheses = []

    test_iterator = test_dataset.as_numpy_iterator()
    batch_count = 0

    for batch in test_iterator:
        if num_batches and batch_count >= num_batches:
            break

        videos, alignments = batch

        # Make predictions
        yhat = model.predict(videos, verbose=0)
        decoded = decode_predictions(yhat)

        # Process each sample in batch
        for i in range(len(videos)):
            # Convert to text
            reference = tokens_to_text(alignments[i])
            hypothesis = tokens_to_text(decoded[i])

            # Clean up text (remove extra spaces)
            reference = ' '.join(reference.split())
            hypothesis = ' '.join(hypothesis.split())

            # Calculate metrics
            cer = calculate_cer(reference, hypothesis)
            wer = calculate_wer(reference, hypothesis)

            all_cer.append(cer)
            all_wer.append(wer)
            all_references.append(reference)
            all_hypotheses.append(hypothesis)

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processed {batch_count} batches...")

    # Calculate overall statistics
    mean_cer = np.mean(all_cer)
    mean_wer = np.mean(all_wer)
    median_cer = np.median(all_cer)
    median_wer = np.median(all_wer)
    std_cer = np.std(all_cer)
    std_wer = np.std(all_wer)

    results = {
        'mean_cer': mean_cer,
        'mean_wer': mean_wer,
        'median_cer': median_cer,
        'median_wer': median_wer,
        'std_cer': std_cer,
        'std_wer': std_wer,
        'num_samples': len(all_cer),
        'all_cer': all_cer,
        'all_wer': all_wer,
        'references': all_references,
        'hypotheses': all_hypotheses
    }

    return results


def print_results(results):
    """Print evaluation results in a nice format"""
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)
    print(f"\nNumber of samples evaluated: {results['num_samples']}")
    print(f"\n{'Metric':<20} {'Mean':<15} {'Median':<15} {'Std Dev':<15}")
    print("-" * 70)
    print(f"{'Character Error Rate':<20} {results['mean_cer']:<15.4f} {results['median_cer']:<15.4f} {results['std_cer']:<15.4f}")
    print(f"{'Word Error Rate':<20} {results['mean_wer']:<15.4f} {results['median_wer']:<15.4f} {results['std_wer']:<15.4f}")
    print("\n" + "=" * 100)

    # Show percentage format
    print(f"\nCER: {results['mean_cer']*100:.2f}% (lower is better)")
    print(f"WER: {results['mean_wer']*100:.2f}% (lower is better)")
    print("\n" + "=" * 100 + "\n")


def show_examples(results, num_examples=5):
    """Show some example predictions"""
    print("\n" + "=" * 100)
    print(f"SHOWING {num_examples} EXAMPLE PREDICTIONS")
    print("=" * 100 + "\n")

    for i in range(min(num_examples, len(results['references']))):
        print(f"Example {i+1}:")
        print(f"  Reference:  {results['references'][i]}")
        print(f"  Hypothesis: {results['hypotheses'][i]}")
        print(f"  CER: {results['all_cer'][i]:.4f} ({results['all_cer'][i]*100:.2f}%)")
        print(f"  WER: {results['all_wer'][i]:.4f} ({results['all_wer'][i]*100:.2f}%)")
        print("-" * 100 + "\n")


# ============================================================================
# 5. Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("LipNet Model Testing Script")
    print("=" * 100 + "\n")

    # Load data
    print("Loading test data...")
    data = tf.data.Dataset.list_files('./data/s5/*.mpg')
    data = data.shuffle(2000, reshuffle_each_iteration=False)
    data = data.map(mappable_function)
    data = data.padded_batch(8, padded_shapes=([75, 46, 140, 1], [40]))
    data = data.prefetch(tf.data.AUTOTUNE)

    # Calculate dataset size
    total_samples = sum(1 for _ in data)
    print(f"Total batches in dataset: {total_samples}")

    # Recreate dataset after counting
    data = tf.data.Dataset.list_files('./data/s5/*.mpg')
    data = data.shuffle(2000, reshuffle_each_iteration=False)
    data = data.map(mappable_function)
    data = data.padded_batch(8, padded_shapes=([75, 46, 140, 1], [40]))
    data = data.prefetch(tf.data.AUTOTUNE)

    # Split into train/test (same as training)
    train_size = int(total_samples * 0.9)
    test = data.skip(train_size)

    print(f"Test batches: {total_samples - train_size}")

    # Build and load model
    print("\nBuilding model...")
    model = build_model()

    print("Loading model weights...")
    try:
        model.load_weights('models/checkpoint_fast.weights.h5')
        print("✓ Model weights loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model weights: {e}")
        print("Please ensure 'models/checkpoint_fast.weights.h5' exists")
        exit(1)

    # Evaluate model
    results = evaluate_model(model, test, num_batches=None)  # Set to None to evaluate all batches

    # Print results
    print_results(results)

    # Show example predictions
    show_examples(results, num_examples=10)

    # Save results to file
    results_file = 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write("LipNet Model Test Results\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Number of samples: {results['num_samples']}\n\n")
        f.write(f"Character Error Rate (CER):\n")
        f.write(f"  Mean:   {results['mean_cer']:.4f} ({results['mean_cer']*100:.2f}%)\n")
        f.write(f"  Median: {results['median_cer']:.4f} ({results['median_cer']*100:.2f}%)\n")
        f.write(f"  Std:    {results['std_cer']:.4f}\n\n")
        f.write(f"Word Error Rate (WER):\n")
        f.write(f"  Mean:   {results['mean_wer']:.4f} ({results['mean_wer']*100:.2f}%)\n")
        f.write(f"  Median: {results['median_wer']:.4f} ({results['median_wer']*100:.2f}%)\n")
        f.write(f"  Std:    {results['std_wer']:.4f}\n\n")
        f.write("=" * 100 + "\n")

    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 100)
    print("Testing Complete!")
    print("=" * 100 + "\n")
