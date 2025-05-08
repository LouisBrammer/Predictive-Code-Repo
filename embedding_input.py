#Load the pretrained embeddings
path_to_glove_file = "data/glove.6B/glove.6B.100d.txt"
embeddings_index = {}
with open(os.path.expanduser(path_to_glove_file)) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")

max_length = 600
max_tokens = 20000

tokenizer = keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=max_length, output_mode="int")

train_dataset_text_only = train_dataset.map(lambda x, y: x)

tokenizer.adapt(train_dataset_text_only)

train_dataset_int = train_dataset.map(lambda x, y: (tokenizer(x), y), num_parallel_calls=4)

valid_dataset_int = valid_dataset.map(lambda x, y: (tokenizer(x), y), num_parallel_calls=4)

embedding_dim = 100

vocabulary = tokenizer.get_vocabulary()

word_index = dict(zip(vocabulary, range(len(vocabulary))))

embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embedding_layer = keras.layers.Embedding(
    max_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=True,
)