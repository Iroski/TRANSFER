import os
import sys
import time
import pickle
import random
import numpy
import tensorflow as tf
from tensorflow import keras,RaggedTensor
from tensorflow.keras.layers import Input,LSTM,Dense,MaxPool1D,Bidirectional
from tensorflow.keras.models import Model
from sklearn import metrics

def load_from_file(file_path):
	with open(file_path, "rb") as file:
		return pickle.load(file)
class EnhancedEmbedding(tf.keras.layers.Embedding):
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None, **kwargs):
        super().__init__(input_dim, output_dim, embeddings_initializer, embeddings_regularizer, activity_regularizer, embeddings_constraint, mask_zero, input_length, **kwargs)
        self.five=tf.constant(0.5)
        self.zero=tf.constant(0)
    
    # @tf.function
    def embedding(self,inputs):
        return super().call(inputs)
    
    # @tf.function
    def map_2(self,tokens):
        identifier=self.embedding(tokens[0])
        cur_word=self.embedding(tokens[2])
        if identifier.shape[0]==self.zero:
                return tf.squeeze(cur_word)
        return tf.squeeze(tf.reduce_mean(identifier)*self.five+cur_word*self.five)
    # @tf.function
    def map_1(self,inputs):
        return tf.map_fn(fn=lambda x :self.map_2(x),elems=inputs,dtype=tf.float32)

    def call(self, inputs):
        final_embeddings=tf.map_fn(fn=lambda x:self.map_1(x),elems=inputs,dtype=tf.float32)

        return final_embeddings
class EnhancedModel(Model):
    def __init__(self,  embedding_dim, hidden_dim, vocab_size, label_size,seq_len, pretrained_weight):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.label_size = label_size
        self.activation = tf.keras.activations.tanh
        self.num_layers = 1

        self.embedding=EnhancedEmbedding(vocab_size,embedding_dim,embeddings_initializer=keras.initializers.Constant(pretrained_weight))
        # self.encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True))
        self.encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True,input_shape=(seq_len,embedding_dim)))
        self.pool=MaxPool1D(hidden_dim*2)
        self.decoder = Dense(self.label_size)

    def call(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings=tf.reshape(embeddings,[-1,400,32])
        # print("finish embedding")
        # print(embeddings.shape)
        lstm_out = self.encoder(embeddings)
        # print("finish lstm")
        # print(lstm_out.shape)
        lstm_out = tf.transpose(lstm_out, perm=[0,2,1])
        # print(lstm_out.shape)
        pool_out=self.pool(lstm_out)
        # print("pool shape")
        # print(pool_out.shape)
        out = tf.squeeze(pool_out,[1])
        # print("finish pool")
        # print(out.shape)
        out = self.decoder(out)
        # print("finish linear")
        return out

def built_model(train_data,val_data,batch_size,embedding_dim, hidden_dim, max_tokens, labels,seq_len, pretrain_weight):
    model = EnhancedModel(embedding_dim, hidden_dim, max_tokens, labels,seq_len, pretrain_weight)
    model.compile(optimizer=tf.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    model.fit(
        train_data,
        steps_per_epoch=batch_size,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback]
        )

    return model
if __name__=='__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    strategy = tf.distribute.get_strategy()
    fix_pattern = 'InsertMissedStmt'
    print("Fix pattern: {}".format(fix_pattern))
    root = "../data/{}/".format(fix_pattern)
    pretrain_vectors = load_from_file(os.path.join(root, "vectors.pkl"))

    AUTO = tf.data.experimental.AUTOTUNE
    SEQ_LEN=2
    HIDDEN_DIM = 50
    EPOCHS = 3
    BATCH_SIZE = 64
    LABELS = 2
    USE_GPU = True
    MAX_TOKENS = pretrain_vectors.shape[0]
    EMBEDDING_DIM = pretrain_vectors.shape[1]

    train_x = load_from_file(os.path.join(root, "train/x_w2v_embed_more.pkl"))
    val_x = load_from_file(os.path.join(root, "val/x_w2v_embed_more.pkl"))
    test_x = load_from_file(os.path.join(root, "test/x_w2v_embed_more.pkl"))
        
    train_y = load_from_file(os.path.join(root, "train/y_.pkl"))
    val_y = load_from_file(os.path.join(root, "val/y_.pkl"))
    test_y = load_from_file(os.path.join(root, "test/y_.pkl"))

    train_x_enc=tf.ragged.constant(train_x,dtype=tf.int32)
    val_x_enc=tf.ragged.constant(val_x,dtype=tf.int32)
    test_x_enc=tf.ragged.constant(test_x,dtype=tf.int32)

    with strategy.scope():
        start_time = time.time()
        tf.config.run_functions_eagerly(True)
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_x_enc, train_y))
                .batch(len(train_y)//BATCH_SIZE)
                .cache()
                .prefetch(AUTO)
        )
        valid_dataset = (
            tf.data.Dataset.from_tensor_slices((val_x_enc, val_y))
                .batch(len(val_y)//BATCH_SIZE)
                .cache()
                .prefetch(AUTO)
        )
        model=built_model(train_dataset,valid_dataset,BATCH_SIZE,EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS, LABELS,SEQ_LEN, pretrain_vectors)
        end_time = time.time()
        print("time cost: %.3f"%(end_time - start_time))
        y_predict=model.predict(test_x_enc)
        result=[]
        for item in y_predict:
            if item[0]>item[1]:
                result.append(0)
            else:
                result.append(1)
        print('准确率：', metrics.accuracy_score(test_y, result))
        print('分类报告:', metrics.classification_report(test_y, result))