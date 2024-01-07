### libraries and imports

import keras
import autokeras as ak
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix
from sklearn.utils import resample
import gc

# constants
#path = 'C:/Users/Asus/Documents/2023_2/Memoria/src'
path = 'C:/Users/Monte/Documents/Memoria/src'

# seeds
global_seed = 19
# np.random.seed(seed=global_seed) # seed gets seted by statement below
keras.utils.set_random_seed(global_seed)
tf.config.experimental.enable_op_determinism()

# Bstar
B_df = pd.read_csv(path+'/Data/b_spectral_lines_trim.csv')
# Observed spectra
Obs_df = pd.read_csv(path+'/Data/observed_spectral_lines_trim_v2.csv')


##### Data formatting #####

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

target = ['temperature','g','v']

X = B_df.drop(target,axis = 1)
Y = B_df[target]
X_obs = Obs_df.drop(target+['HD','date','time'],axis=1)
Y_obs = Obs_df[target]

X = np.random.normal(loc=0, scale=0.01, size=X.shape) + X
y_scaler = MinMaxScaler()

x_train, x_val, y_train, y_val = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=global_seed)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, test_size=0.25, random_state=global_seed)

y_scaler = y_scaler.fit(y_train)
y_train = y_scaler.transform(y_train)
y_val = y_scaler.transform(y_val)
y_test = y_scaler.transform(y_test)
y_obs = y_scaler.transform(Y_obs.values)

formatted_x_train = np.expand_dims(x_train, axis=2) #This step it's very important an CNN will only accept this data shape
formatted_x_val = np.expand_dims(x_val, axis=2)
formatted_x_test = np.expand_dims(x_test, axis=2)
formatted_x_obs = np.expand_dims(X_obs, axis=2)



##### Auto Model #####

from keras_tuner.engine import hyperparameters as hp

# convBlock
conv_kernel_size = hp.Choice("kernel_size", [2,3,5],)
conv_num_layers = hp.Choice("conv_layers", [2,3],)
conv_num_blocks = hp.Choice("conv_blocks", [2,3],)
conv_filters = hp.Choice("filters", [32,64,128,256,512],)

# # DenseBlock
dense_num_layers = hp.Choice("dense_layers", [2,3])
dense_units = hp.Choice("dense_units", [128,256,384,512])

# RegressionHead
# head_dropout = hp.Float(
#                 'dropout_head',
#                 min_value=0.0,
#                 max_value=0.5,
#                 default=0.25,
#                 step=0.05,
#             )

input = ak.Input()
conv = ak.ConvBlock(
  	kernel_size=conv_kernel_size,
  	num_blocks=conv_num_blocks,
  	num_layers=conv_num_layers,
  	filters=conv_filters,
  	max_pooling=True,
		separable=False,
  	#dropout=conv_dropout,
)(input)
dense = ak.DenseBlock(
  	num_layers=dense_num_layers, 
  	num_units=dense_units, 
  	use_batchnorm=False, 
  	#dropout=dense_dropout
)(conv)
regression_output = ak.RegressionHead(
  	output_dim=3, 
   	loss="mse", 
    metrics=["mae"], 
    #dropout=head_dropout,
)(dense)


def generateAutoModel():
  return ak.AutoModel(
	inputs=input,
	outputs=regression_output,
	max_trials=300,
	loss='mse',
	metrics = 'mae',
	tuner='greedy',
	overwrite=False,
	seed=global_seed,
	max_model_size=None,
	directory='./greedy',
 	project_name='greedy_approach')



##### write & read predictions functions

def write_predictions(fp,predictions):
    for instance in predictions:
        t, g, v = instance
        fp.write(str(t)+' '+str(g)+' '+str(v)+'\n')
        
def read_predictions(fp,num_predictions):
    predictions = []
    lines = fp.readlines()
    for i in range(num_predictions):
        t, g, v = lines[i].split()
        predictions.append([float(t),float(g),float(v)])
    return np.array(predictions)



##### Ensemble #####
def __main__():
	i_fp = open('iteration.txt','r')
	i = int(i_fp.readline())
	i_fp.close()
	weights_fp = open('./data/weights.txt','a')
	keras.utils.set_random_seed(i)
    
	x_train_sparse = coo_matrix(x_train)
	x_train_resampled, x_train_sparse, y_train_resampled = resample(x_train, x_train_sparse, y_train, random_state=i)
	x_train_resampled = np.expand_dims(x_train_resampled, axis=2)
    
	am = generateAutoModel()
	am.fit(x=x_train_resampled, y=y_train_resampled, epochs=30)
     
	test_predictions = am.predict(formatted_x_test)
	obs_predictions = am.predict(formatted_x_obs)
	mse, mae = am.evaluate(formatted_x_val, y_val)
	test_predictions_fp = open('./data/test/'+str(i)+'.predictions.txt','w')
	obs_predictions_fp = open('./data/obs/'+str(i)+'.predictions.txt','w')
     
	write_predictions(test_predictions_fp,test_predictions)
	write_predictions(obs_predictions_fp,obs_predictions)
	weights_fp.write(str(mse)+' '+str(mae)+'\n')
     
	test_predictions_fp.close()
	obs_predictions_fp.close()
	weights_fp.close()
     
	tf.keras.backend.clear_session()
	del test_predictions_fp
	del obs_predictions_fp
	gc.collect()
    
__main__()