"""
Assessment 2: e-Commerce Text Classifier
URL: https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification/code
"""

#%%
#1. import package and define hyperparameter
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
#%%
#defining hyperparameter for model training
vocab_size = 5000
embedding_dim = 64
max_length = 400
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8
#%%
#2. data loading
#add "label" and "text" as column header
df = pd.read_csv("ecommerceDataset.csv", header=None)
df.columns = ["label", "text"]
df.head()
# %%
#3. data inspection and cleaning
df.shape
# %%
#check for NaN
df.isna().sum()
# %%
#remove NaN
df.dropna(inplace=True)
# %%
#recheck NaN
df.isna().sum()
# %%
#check for duplicated
df.duplicated().sum()
# %%
#remove duplicated
df.drop_duplicates(inplace=True)
# %%
#recheck for duplicated
df.duplicated().sum()
# %%
#current data shape
df.shape
# %%
#check label
df.label.value_counts()
#%%
#4. feature selection
feature = df["text"].values
label = df["label"].values
#%%
#convert labels into integer
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
#%%
label_process = label_encoder.fit_transform(label)
#%%
#5.train test split dataset
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(feature,label_process,train_size=training_portion,random_state=12345)
#%%
#6.tokenized train feature (X_train)
from tensorflow import keras
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, split=" ", oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
#%%
#print first 10 word_index item
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))
# %%
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# %%
#7. padding and truncate
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_tokens,maxlen=(max_length))
X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_tokens,maxlen=(max_length))

# %%
#8. model development
#create sequential model
model = keras.Sequential()
#add embedding layer as input layer
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#add the bidirectional lstm layer
model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)))
model.add(keras.layers.Dropout(0.5))
#classifiaction layer
model.add(keras.layers.Dense(embedding_dim, activation="relu"))
model.add(keras.layers.Dense(len(np.unique(y_train)),activation="softmax"))

model.summary()

# %%
#9. model compilation
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#%%
#connect to tensorboard
tensorboard = TensorBoard(log_dir='logdir\\{}'.format("assessment_2"))

# %%
#10. model training
history = model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=10, batch_size=128, callbacks=[tensorboard])
#%%
#11. Model Evaluation
print(history.history.keys())

#%%
#plot graph loss vs val_loss to see the trend
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train loss', "Test loss"])
plt.show()
#%%
#12. model deployment
#test result of the model by defining and tokenized new validation input
test_string1 = ["Harry Potter and the philosopher stone"]
test_string_tokens = tokenizer.texts_to_sequences(test_string1)
test_string_padded = keras.preprocessing.sequence.pad_sequences(test_string_tokens, maxlen=(max_length))
#%%
#padded the new validation input
y_pred = np.argmax(model.predict(test_string_padded),axis=0)
#%%
#label_map is arrange by comparing aranggement in label_process and label
label_map = ["Books","Clotihng & Accessories","Electronics","Household"]
predicted_sentiment = [label_map[i] for i in y_pred]
#%%
#13. save model and tokenizer
import os

PATH = os.getcwd()
print(PATH)
#%%
#model save path
from tensorflow.keras.models import load_model

model_save_path = os.path.join(PATH,"ecommerce_text_classifier.h5")
keras.models.save_model(model,model_save_path)
#%%
#load save model
model_loaded = keras.models.load_model(model_save_path)
#%%
#save tokenizer
import pickle
tokenizer_save_path = os.path.join(PATH,"tokenizer.json")
with open(tokenizer_save_path,"wb") as f:
    pickle.dump(tokenizer,f)
#%%
#loaded tokenizer
with open(tokenizer_save_path,"rb") as f:
    tokenizer_loaded = pickle.load(f)
#%%
