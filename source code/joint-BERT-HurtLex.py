!pip install bert-tensorflow==1.0.1
%tensorflow_version 1.x

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import keras as k
from sklearn import metrics
import codecs
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from bert.tokenization import FullTokenizer
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_score, recall_score
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
#bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
bert_path = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"

max_seq_length = 100

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with codecs.open(fp, encoding="utf-8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[1]
                misogyny = int(line.split("\t")[2])
                y.append(misogyny)
                corpus.append(tweet)

    return corpus, y

def hurtLexEmbedding (fp, X):
    embeddings_index = {}
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 15000)
    tokenizer.fit_on_texts(X)
    with codecs.open(fp, encoding='utf-8') as f:
        print("read embedding...")
        for line in f:
            values = line.split("\t")
            word = values[0]
            coefs = np.asarray(values[1].split(" "), dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    print("vocabulary size = "+str(len(tokenizer.word_index)))
    print("embedding size = "+str(len(embeddings_index)))
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 17))
    unk_dict = {}
    vocab = len(tokenizer.word_index)
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        elif word in unk_dict:
            embedding_matrix[i] = unk_dict[word]
        else:
            # random init, see https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
            unk_embed = np.zeros(17, dtype='float32')
            unk_dict[word] = unk_embed
            embedding_matrix[i] = unk_dict[word]
    return vocab, embedding_matrix


dataTrain, dataLabel = parse_training("italian-balanced.tsv")
dataTest, labelTest = parse_training("italian.tsv")  

dataTrain1, dataLabel1 = parse_training("english-balanced.tsv")
dataTest1, labelTest1 = parse_training("italian-english.tsv")  

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()

# Convert data to InputExample format

# Create datasets (Only take up to max_seq_length words for memory)
train_text = [' '.join(t.split()[0:max_seq_length]) for t in dataTrain]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]

train_text1 = [' '.join(t.split()[0:max_seq_length]) for t in dataTrain1]
train_text1 = np.array(train_text1, dtype=object)[:, np.newaxis]
#train_label = train_df['polarity'].tolist()

test_text = [' '.join(t.split()[0:max_seq_length]) for t in dataTest]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]

test_text1 = [' '.join(t.split()[0:max_seq_length]) for t in dataTest1]
test_text1 = np.array(test_text1, dtype=object)[:, np.newaxis]
#test_label = test_df['polarity'].tolist()


train_examples = convert_text_to_examples(train_text, dataLabel)
test_examples = convert_text_to_examples(test_text, labelTest)

train_examples1 = convert_text_to_examples(train_text1, dataLabel1)
test_examples1 = convert_text_to_examples(test_text1, labelTest1)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
(test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

(train_input_ids1, train_input_masks1, train_segment_ids1, train_labels1) = convert_examples_to_features(tokenizer, train_examples1, max_seq_length=max_seq_length)
(test_input_ids1, test_input_masks1, test_segment_ids1, test_labels1) = convert_examples_to_features(tokenizer, test_examples1, max_seq_length=max_seq_length)

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = False
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

# Build model
def build_model(data_nen, data_en, max_seq_length): 
    hurtlex_vocab_nen, hurtlex_matrix_nen = hurtLexEmbedding("it-hurtlex-embedding.vec", data_nen)
    hurtlex_vocab_en, hurtlex_matrix_en = hurtLexEmbedding("hurtlex-embedding.vec", data_en)

    input_sequence = tf.keras.layers.Input(name='inputs_nen',shape=(None,))
    input_sequence1 = tf.keras.layers.Input(name='inputs_en',shape=(None,))

    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    hurtlex_embedding_nen = tf.keras.layers.Embedding(hurtlex_vocab_nen+1, 17, input_length=100, weights=[hurtlex_matrix_nen], trainable=False)(input_sequence)
    bert_inputs = [in_id, in_mask, in_segment]

    in_id1 = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids1")
    in_mask1 = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks1")
    in_segment1 = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids1")
    hurtlex_embedding_en = tf.keras.layers.Embedding(hurtlex_vocab_en+1, 17, input_length=100, weights=[hurtlex_matrix_en], trainable=False)(input_sequence1)
    bert_inputs1 = [in_id1, in_mask1, in_segment1]
    
    bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)
    bert_output1 = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs1)

    lstm = tf.keras.layers.LSTM(32)(hurtlex_embedding_nen)
    lstm1 = tf.keras.layers.LSTM(32)(hurtlex_embedding_en)

    dense_bert1 = tf.keras.layers.Dense(128, activation='relu')(bert_output)
    dense_bert2 = tf.keras.layers.Dense(128, activation='relu')(bert_output1)
    dense_hurtlex1 = tf.keras.layers.Dense(16, activation='relu')(lstm)
    dense_hurtlex2 = tf.keras.layers.Dense(16, activation='relu')(lstm1)

    concat1 = tf.keras.layers.concatenate([dense_bert1, dense_hurtlex1], axis=-1)
    concat2 = tf.keras.layers.concatenate([dense_bert2, dense_hurtlex2], axis=-1)

    concat = tf.keras.layers.concatenate([concat1, concat2], axis=-1)

    pred = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
    
    model = tf.keras.models.Model(inputs=[bert_inputs,bert_inputs1,input_sequence,input_sequence1], outputs=pred)
    myadam = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=myadam, metrics=['accuracy'])
    model.summary()
    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

model = build_model(dataTrain, dataTrain1, max_seq_length)

# Instantiate variables
initialize_vars(sess)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = tf.keras.callbacks.ModelCheckpoint('best_model.ckpt', monitor='val_loss',
                                                 save_weights_only=True,
                                                 verbose=1, save_best_only=True)

max_words = 15000

tok1 = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tok1.fit_on_texts(dataTrain)

tok2 = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tok2.fit_on_texts(dataTrain1)

sequences = tok1.texts_to_sequences(dataTrain)
sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=max_seq_length)

sequences1 = tok2.texts_to_sequences(dataTrain1)
sequences_matrix1 = tf.keras.preprocessing.sequence.pad_sequences(sequences1,maxlen=max_seq_length)

model.fit(
    [train_input_ids, train_input_masks, train_segment_ids,train_input_ids1, train_input_masks1, train_segment_ids1, sequences_matrix, sequences_matrix1], 
    train_labels,
#    validation_split=0.2,
#    verbose=True,
#    callbacks=[es, mc],
    epochs=3,
    batch_size=64
)

test_sequences = tok1.texts_to_sequences(dataTest)
test_sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(test_sequences,maxlen=max_seq_length)

test_sequences1 = tok2.texts_to_sequences(dataTest1)
test_sequences_matrix1 = tf.keras.preprocessing.sequence.pad_sequences(test_sequences1,maxlen=max_seq_length)

y_prob = model.predict([test_input_ids, test_input_masks,  test_segment_ids, test_input_ids1, test_input_masks1,  test_segment_ids1, test_sequences_matrix, test_sequences_matrix1])
#print(y_prob)
y_pred = np.where(y_prob > 0.5, 1, 0)
#print(y_pred)

acc = accuracy_score(labelTest,y_pred)
p0 = precision_score(labelTest,y_pred, pos_label=0)
r0 = recall_score(labelTest,y_pred, pos_label=0)
f0 = f1_score(labelTest,y_pred, pos_label=0)
p1 = precision_score(labelTest,y_pred, pos_label=1)
r1 = recall_score(labelTest,y_pred, pos_label=1)
f1 = f1_score(labelTest,y_pred, pos_label=1)

print(p0)
print(r0)
print(f0)
print(p1)
print(r1)
print(f1)
print(acc)

CorpusFile = './prediction-bert-hurtlex-it.tsv'
CorpusFileOpen = codecs.open(CorpusFile, "w", "utf-8")
for label in y_pred:
    CorpusFileOpen.write(str(label[0])) 
    CorpusFileOpen.write("\n")   
CorpusFileOpen.close()