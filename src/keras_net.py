import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD

################################################################################
### Network definition
################################################################################


model = Sequential([
	Conv2D(16, (3,3), (1,1), padding='same', activation='relu', input_shape=(64,64,1)),
	MaxPooling2D(),
	Conv2D(32, (3,3), (1,1), padding='same', activation='relu'),
	MaxPooling2D(),
	Conv2D(64, (3,3), (1,1), padding='same', activation='relu'),
	Dense(1024),
	Dense(10),
	Activation('softmax')
])


model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])




