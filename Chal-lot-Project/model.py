from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D,Conv2D, ZeroPadding2D
from keras.optimizers import RMSprop
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

batch_size = 64
epochs = 500

# Data Parsing
data = pd.read_csv('./data/merge.csv', header=None);
data = data.reindex(np.random.permutation(data.index));
train, test = train_test_split(data, train_size=0.9, random_state=0);
x_train = train.iloc[:, 1:].values
y_train = np_utils.to_categorical(train.iloc[:, 0].values)
x_test = test.iloc[:, 1:].values
y_test = np_utils.to_categorical(test.iloc[:, 0].values)
print (data.shape)

# Model : Alexnet
# Input : 62 x 30
# Output : 81 x 1 (81 cells)
input_shape = (x_train.shape[1],)
output_shape = y_train.shape[1]

model = Sequential()

model.add(Conv2D(96, kernel_size=(11, 11),
                 activation='relu',
                 input_shape=input_shape))  		# 62 x 30 -> 52 x 20
model.add(Conv2D(256, (5, 5), activation='relu'))  	# 52 x 20 -> 48 x 16
model.add(MaxPooling2D(pool_size=(2, 2))) 		 	# 48 x 16 -> 24 x 8
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu')) 	# 24 x 8 -> 22 x 6
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(1024, (3, 3), activation='relu'))  # 22 x 6 -> 20 x 4
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(1024, (3, 3), activation='relu'))  # 20 x 4 -> 18 x 2
model.add(Flatten())
model.add(Dense(3072, activation='relu'))  			# 3072
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))  			# 4096
model.add(Dropout(0.5))
model.add(Dense(output_shape, activation='softmax'))

model.summary()

# Training
model.compile(loss='categorical_crossentropy',
				optimizer = RMSprop(),
				metrics = ['accuracy'])

history = model.fit(x_train, y_train,
				batch_size = batch_size,
				epochs = epochs,
				verbose = 1,
				validation_data = (x_test, y_test))


# Result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
