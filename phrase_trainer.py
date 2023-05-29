# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
# Define hyperparameters
vocab_size = 150000 # Number of words in the vocabulary
embedding_dim = 128 # Dimension of word embeddings
max_length = 50 # Maximum length of input sequences
trunc_type = 'post' # Truncate sequences after max_length
padding_type = 'post' # Pad sequences after max_length
oov_token = '<OOV>' # Token for out-of-vocabulary words
num_epochs = 15 # Number of training epochs
word_index = None

# Create a tokenizer and fit it on the corpus
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
text_data = open('text_data.txt').read()
corpus = text_data.split('\n') # Split the text into sentences one sentence per newline
# corpus = text_data.replace("?",".").replace("!",".").split('.') # Split the text into sentences
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index

def traind_data():
	# Load the text data (you can use any text corpus you want)

	 # Get the word-index mapping
	# saving word index (dict)
	# print(type(word_index))
	# fff = open("word_index.wi","w")
	# fff.write(json.dumps(word_index))
	# fff.close()

	# Convert the sentences into sequences of integers
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
	model.save('f1.model')

def get_model_F1():
	new_model = tf.keras.models.load_model('f1.model')
	return new_model

def save_model_F2(model):
	model.save_weights('my_weights.h5')# Save the weights
	model_json = model.to_json()# Save the architecture in JSON format
	with open('my_model.json', 'w') as json_file:
		json_file.write(model_json)

def get_model_F2():
	with open('my_model.json', 'r') as json_file:
		model_json = json_file.read()
	new_model = tf.keras.models.model_from_json(model_json)
	new_model.load_weights('my_weights.h5')
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
