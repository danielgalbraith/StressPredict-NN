import pandas as pd
import numpy as np
from pandas import DataFrame, get_dummies
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

f = pd.read_csv('presidents-data-words-january-3-2018.csv')
df = DataFrame(f)
df = df.dropna(subset=['dagalb','nseg','nsyll','nstress','mean'])

early_stop = EarlyStopping(patience=5)

X_cols = ['widx','lexstress','nseg','nsyll','nstress','pos','dep','doc.freq','d.inform.3','corpus.freq','c.inform.3','category']
X = df[X_cols]
y = np.array(to_categorical(df.dagalb))

cat_cols = ['lexstress','pos','dep','category']
scale_cols = ['widx','nseg','nsyll','nstress','doc.freq','d.inform.3','corpus.freq','c.inform.3']

for c in cat_cols:
    dum = pd.get_dummies(X[c], columns=[c], prefix=c)
    X = pd.concat([dum, X], axis=1)
    del(X[c])

scaler = MinMaxScaler()
scaler.fit(X[scale_cols])
X[scale_cols] = scaler.transform(X[scale_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

n_cols = X_train.shape[1]

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,), kernel_constraint=max_norm(4)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, callbacks=[early_stop], validation_split=0.1, batch_size=50)
y_pred = model.predict_classes(X_train)
score = model.evaluate(X_test, y_test, batch_size=50)
print("Accuracy on test set = " + format(score[1]*100, '.2f') + "%")
