# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model


# Sampling frequency
freq = 44100
 
# Recording duration
duration = 5
 
# Start recorder with the given values 
# of duration and sample frequency
print("Recording start: ")
recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)

# Record audio for the given number of seconds
sd.wait()
print("Recording Done!")
# This will convert the NumPy array to an audio
# file with the given sampling frequency
 
# Convert the NumPy array to audio file
wv.write("recording.wav", recording, freq, sampwidth=2)

#running model
model = load_model('audio_classification_model.keras')
target_shape = (128, 128)
classes = ['nonHuman', 'Human']

# Function to preprocess and classify an audio file
def test_audio(file_path, model):
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

    # Make predictions
    predictions = model.predict(mel_spectrogram)

    # Get the class probabilities
    class_probabilities = predictions[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    return class_probabilities, predicted_class_index

# Test an audio file
test_audio_file = './recording.wav'
class_probabilities, predicted_class_index = test_audio(test_audio_file, model)

# Display results for all classes
for i, class_label in enumerate(classes):
  probability = class_probabilities[i]
  #print(f'Class: {class_label}, Probability: {probability:.4f}')

# Calculate and display the predicted class and accuracy
predicted_class = classes[predicted_class_index]
#accuracy = class_probabilities[predicted_class_index]
#print(predicted_class_index)
print(f'The audio is classified as: {predicted_class}')
#print(f'Accuracy: {accuracy:.4f}')