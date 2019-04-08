from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K  
import random
import scipy.io
from scipy.stats import mode
import pdb
import os
import keras
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd
from model_util import *

# Define Initial Parameters
pool_subset = 2000
Experiments = 1
batch_size = 64
nb_classes = 10
beta = 0.5
save_dir = os.getcwd()+ '/test/'
score=0
epochs = 50
num_ensembles=5
acquisition_iterations = 50
nb_MC_samples = 100
Queries = 20
Queries_kind = 5
Queries_mult = Queries/Queries_kind
Queries_num = []
for N in range(Queries_kind+1):
	for M in range(Queries_kind+1-N):
		Queries_num.append([N,M,Queries_kind-N-M])
Queries_num.pop(0)
Queries_num = np.array(Queries_num)
Queries_Prob = np.ones(len(Queries_num))/sum(np.ones(len(Queries_num)))
q_ind = np.random.choice(len(Queries_num),1,p=Queries_Prob)[0]
M1_Queries = int(Queries_num[q_ind][0]*Queries_mult)
M2_Queries = int(Queries_num[q_ind][1]*Queries_mult)
U_Queries = int(Queries_num[q_ind][2]*Queries_mult)

Queries_Prob_save = Queries_Prob.copy()
Q_ind = np.array([q_ind],np.int)

Experiments_All_Accuracy = np.zeros(shape=(acquisition_iterations+1))

# Start the Experiment

# for e in range(Experiments):
for e in range(4,-1,-3):

	loss_memory = []

	print('Experiment Number ', e)

	# Load data from previously made matrix
	(X_train_All_org, y_train_All_org), (X_test_org, y_test_org) = mnist.load_data()

	path = os.getcwd() + '/' + 'mnist_train_idx.mat'
	mat = scipy.io.loadmat(path)
	IND1 = mat['IND_V']

	i_labeled = []
	for i in IND1:
		i_labeled += [i]
        
	X_valid = X_train_All_org[i_labeled]
	y_valid = y_train_All_org[i_labeled]
        
	IND2 = mat['IND_P']

	i_labeled = []
	for i in IND2:
		i_labeled += [i]
        
	X_Pool = X_train_All_org[i_labeled]
	y_Pool = y_train_All_org[i_labeled]

	IND3 = mat['IND_T']

	i_labeled = []
	for i in IND3:
		i_labeled += [i]
        
	X_train = X_train_All_org[i_labeled]
	y_train = y_train_All_org[i_labeled]

	y_valid = y_valid.reshape(y_valid.shape[0],) 
	y_Pool = y_Pool.reshape(y_Pool.shape[0],)
	y_train = y_train.reshape(y_train.shape[0],)

	path = os.getcwd() + '/' + 'mnist_test_idx.mat'
	mat = scipy.io.loadmat(path)
	IND = mat['IND2']

	i_labeled = []
	for i in IND:
		i_labeled += [i]

	X_test = X_test_org[i_labeled]
	y_test = y_test_org[i_labeled]
        
	y_test = y_test.reshape(y_test.shape[0],) 

	X_valid = X_valid.reshape(X_valid.shape[0],28, 28, 1)
	X_Pool = X_Pool.reshape(X_Pool.shape[0],28, 28, 1)
	X_train = X_train.reshape(X_train.shape[0],28, 28, 1)
	X_test = X_test.reshape(X_test.shape[0],28, 28, 1)

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')
	X_Pool = X_Pool.astype('float32')
	X_train /= 255
	X_valid /= 255
	X_Pool /= 255
	X_test /= 255

	Y_test = np_utils.to_categorical(y_test, nb_classes)
	Y_valid = np_utils.to_categorical(y_valid, nb_classes)
	Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)
	Y_train = np_utils.to_categorical(y_train, nb_classes)

	x_pool_All = np.zeros(shape=(1))

	AUC = np.array([])
	Precision = np.array([])
	Recall = np.array([])
	F1 = np.array([])
	report_data = {'class0':[],'class1':[],'class2':[],'class3':[],'class4':[],'class5':[],'class6':[],'class7':[],'class8':[],'class9':[]}

	print('Training Model Without Acquisitions in Experiment', e)

	model = model_mnist(X_train, nb_classes)
	hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))

	score, acc = model.evaluate(X_test, Y_test, verbose=0)

	all_accuracy = acc

	print('Starting Active Learning in Experiment ', e)

	# Start the acquisition iterations

	for acq in range(acquisition_iterations):
		print('POOLING ITERATION', acq)

		pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset))
		X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
		y_Pool_Dropout = y_Pool[pool_subset_dropout]

		# Perform MC dropout, refer to model_utils.py

		Avg_Pi = dropout_process(model, X_Pool_Dropout, nb_MC_samples)

		del model
		K.clear_session()
		
		print('Use Ensemble for Uncertainty Calculation')

		# Perform Ensemble and calculate variation ratio, also refer to model_utils.py

		a_2d = v_from_ensembles(num_ensembles, pool_subset, nb_classes, X_train, Y_train, batch_size, epochs, X_valid, Y_valid, X_Pool_Dropout)

		# Select 2 sets of indices(x_pool_index_T : Only var ratio, x_pool_index : Blending Query)

		x_pool_index_T = a_2d.argsort()[-Queries:][::-1]

		x_pool_index = queries(U_Queries, M1_Queries, M2_Queries, a_2d, Avg_Pi, pool_subset, nb_classes, y_train)

		Pooled_X = X_Pool_Dropout[x_pool_index, :, :, :]
		Pooled_Y = y_Pool_Dropout[x_pool_index]
		Pooled_X_T = X_Pool_Dropout[x_pool_index_T, :, :, :]
		Pooled_Y_T = y_Pool_Dropout[x_pool_index_T]

		delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
		delete_Pool_Y = np.delete(y_Pool, (pool_subset_dropout), axis=0)

		X_train_R = np.concatenate((X_train, Pooled_X), axis=0)
		y_train_R = np.concatenate((y_train, Pooled_Y), axis=0)
		Y_train_R = np_utils.to_categorical(y_train_R, nb_classes)

		X_train_T = np.concatenate((X_train, Pooled_X_T), axis=0)
		y_train_T = np.concatenate((y_train, Pooled_Y_T), axis=0)
		Y_train_T = np_utils.to_categorical(y_train_T, nb_classes)

		# Train each model

		model_R = model_mnist(X_train_R, nb_classes)
		hist_R = model_R.fit(X_train_R, Y_train_R, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))
		score_v_R, acc_v_R = model_R.evaluate(X_valid, Y_valid, verbose=0)

		model_T = model_mnist(X_train_T, nb_classes)
		hist_T = model_T.fit(X_train_T, Y_train_T, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))
		score_v_T, acc_v_T = model_T.evaluate(X_valid, Y_valid, verbose=0)

		# Perform multi-armed-bandit according to validation accuracy

		if acc_v_T>acc_v_R:
			loss_memory.append(1)
			Queries_Prob[q_ind] = Queries_Prob[q_ind]*np.exp(-beta/Queries_Prob[q_ind])
			Queries_Prob = Queries_Prob/sum(Queries_Prob)
			X_train = X_train_T
			y_train = y_train_T
			delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index_T), axis=0)
			delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index_T), axis=0)
			score, acc = model_T.evaluate(X_test, Y_test, verbose=0)
			y_pred_keras = model_T.predict(X_test)
			model = model_T
		else:
			loss_memory.append(0)
			X_train = X_train_R
			y_train = y_train_R
			delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index), axis=0)
			delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index), axis=0)
			score, acc = model_R.evaluate(X_test, Y_test, verbose=0)
			y_pred_keras = model_R.predict(X_test)
			model = model_R

		all_accuracy = np.append(all_accuracy, acc)
		del model_R
		del model_T

		# Delete Pooled images

		X_Pool = np.concatenate((delete_Pool_X, delete_Pool_X_Dropout), axis=0)
		y_Pool = np.concatenate((delete_Pool_Y, delete_Pool_Y_Dropout), axis=0)
		Y_train = np_utils.to_categorical(y_train, nb_classes)

		# Update the probability

		q_ind = np.random.choice(len(Queries_num),1,p=Queries_Prob)[0]
		Q_ind = np.append(Q_ind,q_ind)
		Queries_Prob_save = np.vstack([Queries_Prob_save, Queries_Prob])
		M1_Queries = int(Queries_num[q_ind][0]*Queries_mult)
		M2_Queries = int(Queries_num[q_ind][1]*Queries_mult)
		U_Queries = int(Queries_num[q_ind][2]*Queries_mult)

		# Calculate AUC, Precision, Recall, etc and save the results

		auc_score = roc_auc_score(Y_test, y_pred_keras)
		AUC = np.append(AUC, auc_score)
		print('AUC', auc_score)
		target_names = ['class 0','class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']
		Y_test2 = np.argmax(Y_test,axis=1)
		y_pred_keras2 = np.argmax(y_pred_keras,axis=1)
		report = classification_report(Y_test2,y_pred_keras2,target_names=target_names)
		print(report)
		precision,recall,f1,_=precision_recall_fscore_support(Y_test2, y_pred_keras2, average='weighted')	
		Precision = np.append(Precision,precision)
		Recall = np.append(Recall,recall)
		F1 = np.append(F1,f1)

		i = 0
		lines = report.split('\n')
		for xxx in range(2,12):
			row = {}
			line = lines[xxx]
			row_data = line.split('      ')
			row['precision'] = float(row_data[1])
			row['recall'] = float(row_data[2])
			row['f1_score'] = float(row_data[3])
			row['support'] = float(row_data[4])
			report_data['class'+str(i)].append(row)
			i+=1

		for i in range(nb_classes):
			dataframe = pd.DataFrame.from_dict(report_data['class'+str(i)])
			dataframe.to_csv(save_dir+'classification_report_'+ 'Experiment_'+ str(e)+'_class'+str(i) + '.csv', index = False)

		print('Use this trained model with pooled points for Dropout again')
		print('Saving Results Per Experiment')
		np.savetxt(save_dir+'Pooled_Image_Index_'+ 'Experiment_' + str(e) + '.csv', x_pool_All, delimiter=",")
		np.savetxt(save_dir+ 'Accuracy_Results_'+ 'Experiment_' + str(e) + '.csv', all_accuracy, delimiter=",")
		np.savetxt(save_dir+ 'Loss_Memory_'+'Experiment_'+str(e)+'.csv',loss_memory, delimiter=",")
		np.savetxt(save_dir+'Queries_Prob_'+'Experiment_'+str(e)+'.csv',Queries_Prob_save, delimiter=",")
		np.savetxt(save_dir+ 'Queries_ind_'+ 'Experiment_' + str(e) + '.csv', Q_ind ,delimiter=",")
		np.savetxt(save_dir+ 'AUC_'+ 'Experiment_' + str(e) + '.csv', AUC ,delimiter=",")
		np.savetxt(save_dir+ 'Precision_'+ 'Experiment_' + str(e) + '.csv', Precision ,delimiter=",")
		np.savetxt(save_dir+ 'Recall_'+ 'Experiment_' + str(e) + '.csv', Recall ,delimiter=",")
		np.savetxt(save_dir+ 'F1score_'+ 'Experiment_' + str(e) + '.csv', F1 ,delimiter=",")
		np.savetxt(save_dir+'Train_label_'+ 'Experiment_' + str(e) + '.csv', y_train ,delimiter=",")
























