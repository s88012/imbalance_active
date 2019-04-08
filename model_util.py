from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras
from keras import backend as K
import numpy as np
from scipy.stats import mode
from keras.optimizers import Adam
import pdb

def model_mnist(X_train, nb_classes):

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

def model_cifar10(X_train, nb_classes):

	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model


def dropout_process(model, X_Pool_Dropout, nb_MC_samples):

	MC_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
	learning_phase = True
	MC_samples = [MC_output([X_Pool_Dropout, learning_phase])[0] for _ in range(nb_MC_samples)]
	MC_samples = np.array(MC_samples)
	Avg_Pi = np.mean(MC_samples, axis=0)

	return Avg_Pi

def queries(U, P, E, a_2d, Avg_Pi, pool_subset, nb_classes, y_train):

	x_pool_index = np.array([],np.int)

	if U != 0:
		x_pool_index = np.append(a_2d.argsort()[-U:][::-1], x_pool_index)
		Avg_Pi[x_pool_index,:] = [1]*10

	if P!=0:
		predict_y = []
		count_y = []
		min_idx1 = []
		for i in range(pool_subset):
			predict_y.append(np.argmax(Avg_Pi[i,:]))
		for i in range(nb_classes):
			k = predict_y.count(i)
			if k==0:
				k = 1000000
			count_y.append(k)
		while len(min_idx1)<P:
			more_Queries = P-len(min_idx1)
			min_class = np.argmin(count_y)
			count_y[min_class] = 1000000
			min_idx = np.where(predict_y==min_class)[0]
			tmp = Avg_Pi[min_idx,:]
			tmp1 = tmp[:,min_class]
			zipped = zip(min_idx,tmp1)
			XX = sorted(zipped, key = lambda t: t[1])
			XX = XX[:more_Queries]
			ind,val = zip(*XX)
			min_idx1+=ind
		x_pool_index = np.append(min_idx1,x_pool_index)
		Avg_Pi[x_pool_index,:] = [1]*10

	if E!=0:
		predict_y = []
		count_y = []
		min_idx2 = []
		for i in range(pool_subset):
			predict_y.append(np.argmax(Avg_Pi[i,:]))
		for i in range(nb_classes):
			k = list(y_train).count(i)
			count_y.append(k)
		while len(min_idx2)<E:
			more_Queries=E-len(min_idx2)
			while True:
				min_class = np.argmin(count_y)
				count_y[min_class] = 1000000
				min_idx = np.where(predict_y==min_class)[0]
				tmp = Avg_Pi[min_idx,:]
				if len(tmp)!=0:
					break
			tmp1 = tmp[:,min_class]
			zipped = zip(min_idx,tmp1)
			YY = sorted(zipped, key = lambda t: t[1])
			YY = YY[:more_Queries]
			ind, val = zip(*YY)
			min_idx2+=ind
		x_pool_index = np.append(min_idx2, x_pool_index)

	return x_pool_index

def v_from_ensembles(num_ensembles, pool_subset, nb_classes, X_train, Y_train, batch_size, epochs, X_valid, Y_valid, X_Pool_Dropout):

	pred_matrix = np.zeros((num_ensembles, pool_subset))
	Variation = np.zeros((pool_subset))

	for n in range(num_ensembles):
		model_E = model_cifar10(X_train, nb_classes)
		model_E.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))

		pred_matrix[n] = model_E.predict_classes(X_Pool_Dropout)

		del model_E
		K.clear_session()

	for t in range(pool_subset):
		Predicted_Class, Mode = mode(pred_matrix[:,t])
		v = np.array([1-Mode/float(num_ensembles)])
		Variation[t] = v
		
	a_2d = Variation.flatten()

	return a_2d


