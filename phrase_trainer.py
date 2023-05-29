# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json



# Define hyperparameters
vocab_size = 1000 # Number of words in the vocabulary
embedding_dim = 64 # Dimension of word embeddings
max_length = 20 # Maximum length of input sequences
trunc_type = 'post' # Truncate sequences after max_length
padding_type = 'post' # Pad sequences after max_length
oov_token = '<OOV>' # Token for out-of-vocabulary words
num_epochs = 20 # Number of training epochs

# Define hyperparameters
# vocab_size = 10000* # Number of words in the vocabulary
# embedding_dim = 64* # Dimension of word embeddings
# max_length = 10* # Maximum length of input sequences
# trunc_type = 'post' # Truncate sequences after max_length
# padding_type = 'post' # Pad sequences after max_length
# oov_token = '<OOV>' # Token for out-of-vocabulary words
# num_epochs = 10 # Number of training epochs




# =======MODELS==========
TRAIN_DATA = 'train_data/text_data.txt'
MODEL_ = 'models/f1.model'
MODEL_WEIGHTS = 'models/my_weights.h5'
MODEL_WEIGHTS_JSON = 'models/my_model.json'




# Create a tokenizer and fit it on the corpus
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
text_data = open(TRAIN_DATA).read()
corpus = text_data.split('\n') # Split the text into sentences one sentence per newline
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index





def traind_data():
	sequences = tokenizer.texts_to_sequences(corpus)

	# Create input and target sequences by shifting the sequences by one word
	input_sequences = []
	target_sequences = []
	for sequence in sequences:
		for i in range(1, len(sequence)):
			input_sequences.append(sequence[:i])
			target_sequences.append(sequence[i])

	# Pad the input and target sequences to have the same length
	input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
	target_sequences = np.array(target_sequences)

	# Create the RNN model
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_dim, input_length=max_length)) # Embedding layer to learn word embeddings
	model.add(LSTM(128, return_sequences=True)) # LSTM layer to learn sequential patterns
	model.add(LSTM(64)) # Another LSTM layer
	model.add(Dense(vocab_size, activation='softmax')) # Dense layer with softmax activation to output probabilities
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile the model with loss and optimizer
	model.summary() # Print the model summary

	# Train the model on the input and target sequences
	model.fit(input_sequences, target_sequences, epochs=num_epochs, verbose=1)

	save_model_F1(model)
	save_model_F2(model)

# ========================================
def save_model_F1(model):
	model.save(MODEL_)

def get_model_F1():
	new_model = tf.keras.models.load_model(MODEL_)
	return new_model

def save_model_F2(model):
	model.save_weights(MODEL_WEIGHTS)# Save the weights
	model_json = model.to_json()# Save the architecture in JSON format
	with open(MODEL_WEIGHTS_JSON, 'w') as json_file:
		json_file.write(model_json)

def get_model_F2():
	with open(MODEL_WEIGHTS_JSON, 'r') as json_file:
		model_json = json_file.read()
	new_model = tf.keras.models.model_from_json(model_json)
	new_model.load_weights(MODEL_WEIGHTS)
	return new_model

# ========================================================
# Define a function to generate text based on a seed phrase
def generate_text(seed_text,word_count=1):
	model = get_model_F1()
	# model = get_model_F2()
	orig_word = seed_text

	# Convert the seed text into a sequence of integers
	for nums in range(word_count):
		seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
		# Pad the seed sequence
		seed_sequence = pad_sequences([seed_sequence], maxlen=max_length, padding=padding_type, truncating=trunc_type)
		# Predict the next word using the model
		predicted_word_index = np.argmax(model.predict(seed_sequence))
		# Get the predicted word from the word-index mapping
		predicted_word = tokenizer.index_word[predicted_word_index]
		# Return the seed text with the predicted word appended
		seed_text += predicted_word + " "
	return seed_text


traind_data()

# Test the text generation function with some seed phrases
print(generate_text("i'm gonna live my ",10))
