
# coding: utf-8

# # Training of DeepRisk model
# 
# Network uses all covariates:
# 
# * Structural MRI
# * Gender
# * Age
# * ApoE-$\epsilon$4


# In[5]:
from __future__ import print_function
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.layers import Input, concatenate, multiply, add, Reshape, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

sys.path.insert(0, "/trinity/home/jyu/yjvenv/lib/python3.7/site-packages")
import pandas as pd
import os
import numpy as np
import nibabel as nib
import math
import random
import gc

from lifelines.utils import concordance_index
from scipy import ndimage as nd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import argparse


# # 1. Initiliazing

# ## 1.1 Directories

# In[6]:

#folder locations
DATASET_DIR = '/data/scratch/jyu/DeepSurvival/data/'
IMAGE_DIR='/data/scratch/jyu/VBM/'
ADNI_DIR = '/data/scratch/jyu/DeepSurvival/data/ADNI/'
MASK_DIR = '/data/scratch/jyu/DeepSurvival/data/standards/'
MODEL_DIR = '/trinity/home/jyu/DeepSurvival/models/'
#UKBB_DIR = '/lustre5/0/emc17610/Data/UKBB/results/VBM/fsl_mod_files/'
#UKBB_DEMOGRAPHIC_DIR = '/lustre5/0/emc17610/Research/DeepSurvival/data/'
WANG_DIR = '/trinity/home/jyu/BrainAge/'


patients_df = []        #TODO: Unglobalize variable
imgCropping = None


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument("-n",required=True, type=str, help="data file name")
    parser.add_argument("-i",required=True, type=str, help="split number")
    args = parser.parse_args()
    
    
class LoadData:
    """
    Loading preprocessed data from .h5 file.
    Has to be similar to saving data function in data processing notebook.
    (Same names for datasets etc.)
    
    """
    def __init__(self, name):
        dataset_file = name+'.h5'
        
        f = h5py.File(DATASET_DIR+dataset_file, 'r')
#         f = h5py.File('/home/gennadyr/IPython/tests_john/version_age_3/models/'+dataset_file, 'r')
        
        self.fraction_train = f['fraction_train'][:]
        self.fraction_validation = f['fraction_validation'][:]
        self.fraction_test = f['fraction_test'][:]

        self.train_MRI_data = f['train_MRI_data'][:]
        self.validation_MRI_data = f['validation_MRI_data'][:]
        self.test_MRI_data = f['test_MRI_data'][:]
        
        gene_columns = f['gene_column_names'][:]
        
        train_gene_data = f['train_gene_data'][:]
        validation_gene_data = f['validation_gene_data'][:]
        test_gene_data = f['test_gene_data'][:]
        
        self.train_gene_data = pd.DataFrame(train_gene_data, columns=gene_columns)
        self.validation_gene_data = pd.DataFrame(validation_gene_data, columns=gene_columns)
        self.test_gene_data = pd.DataFrame(test_gene_data, columns=gene_columns)
        
        train_label_data1 = f['train_label_data1'][:]
        train_label_data2 = f['train_label_data2'][:]
        validation_label_data1 = f['validation_label_data1'][:]
        validation_label_data2 = f['validation_label_data2'][:]
        test_label_data1 = f['test_label_data1'][:]
        test_label_data2 = f['test_label_data2'][:]
        
        columns = f['label_column_names'][:]
        
        train_label_data2 = pd.DataFrame(train_label_data2)
        train_label_data2.columns = columns
        validation_label_data2 = pd.DataFrame(validation_label_data2)
        validation_label_data2.columns = columns
        test_label_data2 = pd.DataFrame(test_label_data2)
        test_label_data2.columns = columns
        
        train_label_data2['bigrfullname'] = train_label_data1
        validation_label_data2['bigrfullname'] = validation_label_data1
        test_label_data2['bigrfullname'] = test_label_data1

        self.train_label_data = train_label_data2
        self.validation_label_data = validation_label_data2
        self.test_label_data = test_label_data2

        f.close()

        print('Loaded datasets from '+DATASET_DIR+dataset_file)

# ****: Training and Loss class

# In[8]:

class LossHistory(keras.callbacks.Callback):
    """
    Custom callback to show accuracy and loss statistics
    
    """
    def __init__(self, epochs, version):
        self.ne = epochs
        self.mv = version
    
    def on_train_begin(self, logs={}):
        self.validation = 0 # variable to check if validation has started
        self.epoch_num = 0
        self.batch_num = 0
        self.batch_losses = []
        self.epoch_losses = []
        self.epoch_vallosses = []
        self.epoch_cindex = []
        
        print('Start training ...')
        
        self.stats = ['loss'] #TODO: check
        self.logs = [{} for _ in range(self.ne)]

        self.evolution_file = 'val_progress_'+self.mv+'.csv'
        with open(MODEL_DIR+self.evolution_file, "w") as f:
            f.write('val_loss ; val_c\n')
        
        self.progress_file = 'training_progress_'+self.mv+'.out'
        with open(MODEL_DIR+self.progress_file, "w") as f:
            f.write('Start training ...\n')
            
    def on_batch_end(self, epoch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        
        with open(MODEL_DIR+self.progress_file, "a") as f:
            f.write('  >> batch {} >> loss:{}\r'.format(self.batch_num, self.batch_losses[-1]))
        
        self.batch_num += 1
            
 
    
    def on_epoch_end(self, epoch, logs={}):
        self.batch_num = 0
        self.epoch_losses.append(logs.get('loss'))
        ## Custom evaluation metric: C-index, only use in validation
        print('\nCalculating C-index:')
        y_predict = np.asarray(model.predict(data_generator(validation_MRI_set, img_size, batch_size, img_scale, mask, mode='validate', shuffle=False), steps=validation_steps, verbose=0))
        event = []
        y_true = []
        for patient in validation_MRI_set:
            info = validation_label_set.loc[patient]
            event.append(info['dementia'])
            y_true.append(info['event_time'])
        
        # Compute C-index on validation data
        c = 1-concordance_index(y_true, y_predict, event)
        self.epoch_cindex.append(c)
        print('\n    >>> C-index: ', self.epoch_cindex[-1])
        logs['c_index'] = c
        
        # Make riskset
        label_riskset = generate_riskset(np.array(y_true))
        label_data = np.array([event])
        label_riskset = np.array(label_riskset)
        label_data = label_data.transpose()
        y_true1= np.hstack((label_data ,label_riskset))
    
        val_loss=np.mean(CoxPH_loss(y_true1, y_predict))
        self.epoch_vallosses.append(val_loss)
        print('\n    >>> Val_loss: ', self.epoch_vallosses[-1])
        logs['val_loss'] = val_loss
        

        with open(MODEL_DIR+self.progress_file, "a") as f:
            f.write('  >> epoch {} >> c-index:{}\r'.format(self.epoch_num, self.epoch_cindex[-1]))
            
        self.logs[epoch] = logs
        evolution_file = 'evolution_'+self.mv+'.csv'
        loss_fig = 'loss_'+self.mv+'.png'
        self.epoch_num +=1
        
        with open(MODEL_DIR+self.evolution_file, "a") as myfile:
            num_stats = len(self.stats)

            plt.figure(figsize=(25,6))
            plt.suptitle('Model:'+self.mv+ '\ntraining metrics', fontweight='bold')
            
            last_loss = []
            losses = [self.logs[e]['loss'] for e in range(epoch+1)]
            last_loss.append('{}'.format(losses[-1]))
            val_losses=self.epoch_vallosses
            c_indices = self.epoch_cindex
            
            myfile.write(str(val_losses[-1]))
            myfile.write(str(c_indices[-1])+'\n ')
            
            plt.subplot(131)
            plt.plot(range(0,epoch+1),losses, label='Loss')
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.subplot(132)
            plt.plot(range(0,epoch+1),val_losses)
            plt.title('Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Val_Loss')
            plt.subplot(133)
            plt.plot(range(0,epoch+1),c_indices)
            plt.title('Validation C-index')
            plt.xlabel('Epochs')
            plt.ylabel('C-index')
            
            try:                
                plt.savefig(MODEL_DIR+loss_fig)
            except Exception as inst:
                print(type(inst))
                print(inst)
            plt.close()

        with open(MODEL_DIR+self.progress_file, "a") as f:
            f.write('epoch {}/{}:\n'.format(epoch, self.ne))
            f.write('loss = {}\n '.format(last_loss))



# **imgZeropad**: Crops the zero-margin of a 3D image

# In[9]:

class imgZeropad:
    """
    
    
    """
    def __init__(self, img, use_padding=False):
        self.set_crop(img, use_padding)
    
    #set crop locations
    def set_crop(self, img, use_padding=False):
        # argwhere will give you the coordinates of every non-zero point
        true_data = np.argwhere(img)
        # take the smallest points and use them as the top left of your crop
        top_left = true_data.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_data.max(axis=0)
        crop_indeces = [top_left, bottom_right+1]  # plus 1 because slice isn't inclusive

        print('crop set to x[{}:{}], y[{}:{}], z[{}:{}]'.format(crop_indeces[0][0], crop_indeces[1][0], 
                                                                crop_indeces[0][1], crop_indeces[1][1], 
                                                                crop_indeces[0][2], crop_indeces[1][2]))

        if use_padding == True:
            shape = crop_indeces[1]-crop_indeces[0]
            bottom_net = shape.astype(float)/2/2**3
            top_net = np.ceil(bottom_net)*2*2**3
            padding = (top_net-shape)/2
            print('applying [{},{},{}] padding to image..'.format(padding[0], padding[1], padding[2]))
            padding_l = padding.astype(int)
            padding_r = np.ceil(padding).astype(int)
            crop_indeces[0] -= padding_l
            crop_indeces[1] += padding_r

            print('crop set to x[{}:{}], y[{}:{}], z[{}:{}]'.format(crop_indeces[0][0], crop_indeces[1][0], 
                                                                    crop_indeces[0][1], crop_indeces[1][1], 
                                                                    crop_indeces[0][2], crop_indeces[1][2]))
        else:
            padding = np.zeros(3)
        self.crop_indeces = crop_indeces
        self.padding = padding
        
        shape = crop_indeces[1]-crop_indeces[0]
        self.img_size = (shape[0], shape[1], shape[2])

    #crop according to crop_indeces
    def zerocrop_img(self, img, augment=False):
        if augment:
            randx = np.random.rand(3)*2-1
            new_crop = self.crop_indeces+(self.padding*randx).astype(int)

            cropped_img = img[new_crop[0][0]:new_crop[1][0],  
                              new_crop[0][1]:new_crop[1][1],
                              new_crop[0][2]:new_crop[1][2]]

            flip_axis = np.random.rand(3)
            if round(flip_axis[0]):
                cropped_img = cropped_img[::-1,:,:]
            if round(flip_axis[1]):
                cropped_img = cropped_img[:,::-1,:]
            if round(flip_axis[2]):
                cropped_img = cropped_img[:,:,::-1]
                
        else:
            cropped_img = img[self.crop_indeces[0][0]:self.crop_indeces[1][0],  
                              self.crop_indeces[0][1]:self.crop_indeces[1][1],
                              self.crop_indeces[0][2]:self.crop_indeces[1][2]]
            
        return cropped_img


# ### 1.3.1 CNN MRI data input functions

# In[66]:

def retrieve_data(patient_index, img_size, img_scale=1.0, mask=None, augment=False, mode=[]):
    """
    Function to retrieve data from a single patient
    
    Inputs:
    - patient_index = list of bigrfullnames identifying scans
    - img_size = size of MRI images
    - img_scale = scale of the MRI scans [default = 1]
    - mask = mask image if necessary [default = None]
    - augment = Boolean if data augmentation should be used [default = False]
    - mode = train, validate or test (used to find appropriate data)
    
    Outputs:
    - img_data = MRI data
    - input2 = sex of patient
    - input3 = age of patient
    - input4 = ApoE-e4 carriership
    - label = dementia_label (event=1, no event=0)
    - time = event time 
    - genomics = snp carriership for individual 

    """
    # Retrieve patient info and label(=SNP) of the patient
    if mode == 'train':
        patient_info = train_label_set.loc[patient_index]
        ergoid = patient_info.get('ergoid')
    elif mode == 'validate':
        patient_info = validation_label_set.loc[patient_index]
        ergoid = patient_info.get('ergoid')
    elif mode == 'test':
        patient_info = test_label_set.loc[patient_index]
        ergoid = patient_info.get('ergoid')
    else: # validation set might not use validation flag
        patient_info = validation_label_set.loc[patient_index]
        ergoid = patient_info.get('ergoid')
    
    # Get patient label (incident dementia or not)
    label = patient_info.get('dementia')
    
    # Get second input (sex)
    input2 = patient_info.get('sex')
    
    # Get third input (age)
    input3 = patient_info.get('age')
    
    # Get event time
    time = patient_info.get('event_time')

    # Get image
    if ergoid > 0:
        patient_filename = patient_info.name.strip()+'_aseg_GM_to_template_GM_mod.nii.gz'
        img = nib.load(IMAGE_DIR+patient_filename)
            
    elif ergoid < -10000:
        patient_filename = patient_info.name.strip()+'_aseg_GM_to_template_GM_mod.nii.gz'
        img = nib.load(ADNI_DIR+patient_filename)        
        
    img_data = img.get_data()
    # Apply mask to imagedata (if requested)
    if mask is not None:
#        img_data = img_data*mask+(mask-1.0)
        img_data = imgCropping.zerocrop_img(img_data*mask, augment)
    
    # Rescale imagedata (if requested)
    if img_scale < 1.0:
        img_data = resize_img(img_data, img_size)
    
    return np.array(img_data), np.array(int(input2)), np.array(input3), label, time


# In[58]:

def generate_batch(patients, img_size, img_scale=1.0, mask=None, augment=False, mode=[]):
    """
    iterate through a batch of patients and get the corresponding data
    
    Input: 
    - patients = list of bigrfullnames identifying scans
    - img_size = size of MRI images
    - img_scale = scale of the MRI scans [default = 1]
    - mask = mask image if necessary [default = None]
    - augment = Boolean if data augmentation should be used [default = False]
    - mode
    
    Outputs:
    - [input data] = covariates
    - [label data] = label (incident dementia or not) and riskset of patient

    """    
    #get data of each patient
    img_data = []
    label_data = []
    time = []
    sex = []
    age=[]

    for patient in patients:
        try:
            x, x2, x3, y, t = retrieve_data(patient, img_size, img_scale, mask, augment, mode)
            img_data.append(x)
            sex.append(x2)
            age.append(x3)
            label_data.append(y)
            time.append(t)
        except KeyError as e:
            print('\nERROR: No label found for file {}'.format(patient))
        except IOError as e:            
            print('\nERROR: Problem loading file {}. File probably corrupted.'.format(patient))
            
    # Make riskset
    label_riskset = generate_riskset(np.array(time))
    #convert to correct input format for network
    img_data = np.array(img_data)
    img_data = np.reshape(img_data,(-1, 160, 192, 144, 1))

    sex_data = np.array(sex)
    age_data = np.array(age)
    
    label_data = np.array([label_data])
    label_riskset = np.array(label_riskset)
    
    label_data = label_data.transpose()
    label_data_out = np.hstack((label_data,label_riskset))

    return ([img_data, sex_data, age_data], [label_data_out])



# In[59]:

def data_generator(patient_list, img_size, batch_size, img_scale=1.0, mask=None, augment=False, mode=[], shuffle=True):
    """
    Provides the inputs and the label to the convolutional network during training
    
    Input:
    - patient_list = list of bigrfullnames identifying scans
    - img_size = size of MRI images
    - batch_size = size of batch used in training
    - img_scale = scale of the MRI scans [default = 1]
    - mask = mask image if necessary [default = None]
    - augment = Boolean if data augmentation should be used [default = False]
    
    Output:
    - Data = continous data output for batches used in training the network

    """
    while 1:
        if shuffle:
            #shuffle list/order of patients
            pl_shuffled = random.sample(patient_list, len(patient_list))
            #divide list of patients into batches
            batch_size = int(batch_size)
            patient_sublist = [pl_shuffled[p:p+batch_size] for p in range(0, len(pl_shuffled), batch_size)]
        else:
            batch_size = int(batch_size)
            patient_sublist = [patient_list[p:p+batch_size] for p in range(0, len(patient_list), batch_size)]
        count = 0
        data = []
        for batch in range(0, len(patient_sublist)):         
            #get the data of a batch samples/patients
            data.append(generate_batch(patient_sublist[batch], img_size, img_scale, mask, augment, mode))
            count = count + len(patient_sublist[batch])
            #yield the data and pop for memory clearing
            yield data.pop()




# ### 1.3.3 Loss function

# \begin{equation*}
# \large
# Loss = \sum_{i:event=1}(h_{\theta}(x_i) - log(\sum_{j:T_j \geq T_i}e^{h_{\theta}(x_j)})
# \tag{1.1}
# \end{equation*}

# **Function to calculate riskset for batches**

# In[35]:

def generate_riskset(event_times):
    """
    Generates the riskset for every individual. Riskset is the set of individuals that have a 
    longer event time and are thus at risk of experiencing the event : Tj>=Ti
    
    Input:
    - label_data = dataframe with file name, event times and other labels that do not get used
    
    Output:
    - riskset = square matrix in which row i is the riskset of individual i compared to all 
    individuals j. Entry is true if Tj>=Ti, so individual j is 'at risk'.
    """

    o = np.argsort(-event_times, kind="mergesort")
    n_samples = len(event_times)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = event_times[i_sort]
        k = i_org
        while k < n_samples and ti == event_times[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set


# **Function to normalize risk scores**

# In[36]:

def safe_normalize(x):
    """Normalize risk scores to avoid exp underflowing.

    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm


# **Function to calculate log of sum of exponent of predictions** (right hand side of equation 1.1)

# In[37]:

def logsumexp_masked(risk_scores, mask, axis = 0, keepdims= None):
    """
    Computes the log of the sum of the exponent of the predictions across `axis` 
    for all entries where `mask` (riskset) is true:
    
    log(sum(e^h_j))
    
    where h_j are the predictions of patients at risk of developing dementia (T_j>=T_i)
    
    Inputs:
    - risk_scores = the predictions from the network of patients h_j
    - mask = a mask to select which patients are at risk
    
    Output:
    - output = right hand part of the NPLL (Negative Partial Log Likelihood)

    """
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        risk_scores = tf.cast(risk_scores,tf.float32)
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)

        #for numerical stability, substract the maximum value
        #before taking the exponential
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax
        exp_masked = tf.math.multiply(tf.math.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        
        #turn 0's to 1's to get rid of inf loss (log(0) = inf)
        condition = tf.not_equal(exp_sum, 0)   
        exp_sum_clean = tf.where(condition, exp_sum, tf.ones_like(exp_sum))

        output = amax + tf.math.log(exp_sum_clean)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)

    return output


# **Custom loss function** (Negative Partial Log Likelihood)

# In[38]:

def CoxPH_loss(y_true, y_pred):
    """
    Calculates the Negative Partial Log Likelihood:
    
    L = sum(h_i - log(sum(e^h_j)))
    
    where;
        h_i = risk prediction of patient i
        h_j is risk prediction of patients j at risk of developing dementia (T_j>=T_i)
    
    Inputs: 
    - y_true = label data composed of 
        y_event: A 1 or 0 indicating if the patient developed dementia or not, and
        y_riskset:(set of patients j which are at risk dependent on patient i (Tj>=Ti)). 
        
    - y_pred = the risk prediction of the network
    
    Output:
    - loss = the loss used to optimize the network
    
    """
    event = y_true[:,0]
    event = tf.reshape(event,(-1,1))

    riskset_loss = y_true[:,1:]
    predictions = y_pred
    predictions = tf.cast(predictions,tf.float32)
    riskset_loss = tf.cast(riskset_loss,tf.bool)
    event = tf.cast(event, predictions.dtype)
    predictions = safe_normalize(predictions)

#    with tf.name_scope("assertions"):
#        assertions = (
#            tf.debugging.assert_less_equal(event, 1.),
#            tf.debugging.assert_greater_equal(event, 0.),
#            tf.debugging.assert_type(riskset_loss, tf.bool)
#        )

    # move batch dimension to the end so predictions get broadcast
    # row-wise when multiplying by riskset
    pred_t = tf.transpose(predictions)
    # compute log of sum over risk set for each row
    rr = logsumexp_masked(pred_t, riskset_loss, axis=1, keepdims=True)
#    print(predictions.shape.as_list())
#    assert rr.shape.as_list() == predictions.shape.as_list()

    loss = tf.math.multiply(event, rr - predictions)
    
    return loss


# ### 1.3.4 CNN model

# **Network Architecture** 

def pretrained_model(input_shape):
    """
    Model as used in [1], with attached Cox risk prediction layer.
    Using pretrained convolutional layers from [1] and adding newly initialized dense layers.
    
    Model inputs:
     - MRI
     - Genetics
     - Sex
   
    [1] Wang, J., Knol, M. J., ... & Roshchupkin, G. V. (2019). 
    Gray matter age prediction as a biomarker for risk of dementia. PNAS, 116(42), 21213-21218.
    """
    model = tf.keras.models.load_model(WANG_DIR+'model_age_5h.h5')

    input1 = Input(input_shape)# MRI input
    x = input1
    i=0
    for layer in model.layers[1:]: # loop over convolutional layers from [1] and add to model
        if i<28:
            # Freeze the layer (optional)
            #layer.trainable = False  
            # connect the layers
            x = layer(x)
        i+=1

    #right input branch ------------------------
    input2 = Input((1,)) # Sex input
    input3 = Input((1,)) # age input
    
    #merging braches into final model ----------
    x1 = Dense(4, activation='relu')(x)
    y = concatenate([x1, input2, input3])   
    
#     y = concatenate([input2, input3])   
#     y1 = Dense(80, activation='relu')(y)
#     y = concatenate([x, y1]) 
    
    y = Dense(32, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(4, activation='relu')(y)
    final = Dense(1, activation='linear')(y)

    model = Model(inputs=[input1, input2, input3], outputs=final)

    adam_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss=CoxPH_loss, optimizer=adam_opt)
    return model



# # 2. Main code

# ## 2.1 Initializations

# In[46]:

print('--- Starting initialization ---')

#choose variables
use_padding = True
augment_train = True
img_scale = 1.0

percentage_training = 0.7
percentage_validation = 0.15
percentage_testing = 0.15

mask_file = 'Brain_GM_mask_1mm_MNI_kNN_conservative.nii.gz' 

#setup image_size
if mask_file is not None:
    mask = nib.load(MASK_DIR+mask_file).get_data()
    # When applying a mask, initialize zerocropping
    imgCropping = imgZeropad(mask, use_padding=use_padding)
    img_size = np.array(imgCropping.img_size)
else:
    mask = None
    img_size = np.array(np.array(nib.load(IMAGE_DIR+os.listdir(IMAGE_DIR)[0]).get_data()).shape)

img_size = [int(math.ceil(img_d)) for img_d in img_size*img_scale]
print('data shape:', img_size)


# # 3. Prepare data and network

# ## 3.1 Prepare data

# In[47]:

print('--- Preparing datasets and CNN model ---')

print('Keras backend: '+keras.backend.backend())

#make a list of patients numbered from 0 to last
data = LoadData(args.n+'_'+args.i)

#prep full RS datasets
# Train
train_MRI_set = data.train_MRI_data
train_MRI_set=np.char.decode(train_MRI_set)

train_label_set = data.train_label_data
b=train_label_set['bigrfullname'].str.decode("utf-8")
train_label_set['bigrfullname']=1
train_label_set['bigrfullname']=b
train_label_set = train_label_set.set_index('bigrfullname')
train_label_set.columns=train_label_set.columns.str.decode("utf-8")
train_label_set['ergoid']=train_label_set['ergoid'].astype('int')

columnsname=['PRS','age']
mean=train_label_set[columnsname].mean()
std=train_label_set[columnsname].std()
train_label_set[columnsname]=(train_label_set[columnsname]-mean)/std


# Validation
validation_MRI_set = data.validation_MRI_data
validation_MRI_set=np.char.decode(validation_MRI_set)

validation_label_set = data.validation_label_data
b=validation_label_set['bigrfullname'].str.decode("utf-8")
validation_label_set['bigrfullname']=1
validation_label_set['bigrfullname']=b
validation_label_set = validation_label_set.set_index('bigrfullname')
validation_label_set.columns=validation_label_set.columns.str.decode("utf-8")
validation_label_set['ergoid']=validation_label_set['ergoid'].astype('int')

validation_label_set[columnsname]=(validation_label_set[columnsname]-mean)/std



# Test
test_MRI_set = data.test_MRI_data
test_MRI_set=np.char.decode(test_MRI_set)

test_label_set = data.test_label_data
b=test_label_set['bigrfullname'].str.decode("utf-8")
test_label_set['bigrfullname']=1
test_label_set['bigrfullname']=b
test_label_set = test_label_set.set_index('bigrfullname')
test_label_set.columns=test_label_set.columns.str.decode("utf-8")
test_label_set['ergoid']=test_label_set['ergoid'].astype('int')

test_label_set[columnsname]=(test_label_set[columnsname]-mean)/std



#print info per set
train_size = len(train_MRI_set)
print('train samples: {}'.format(train_size))
#train_distr = get_distr_by_list(train_set, subpop_ages, True)[0]
########
# Maybe add sitributions

validation_size = len(validation_MRI_set)
print('validation samples: {}'.format(validation_size))
#validation_distr = get_distr_by_list(validation_set, subpop_ages, True)[0]

test_size = len(test_MRI_set)
print('test samples: {}'.format(test_size))
#test_distr = get_distr_by_list(test_set, subpop_ages, True)[0]

print('\nTotal dataset samples {}'.format(train_size+validation_size+test_size))


# ## 3.2 Initialize network

# In[48]:

#model = cnn_model((160, 192, 144, 1))
model = pretrained_model((160, 192, 144, 1))
model.summary()
model_file = 'model_MRI_Gender_Age_FS.json'
# serialize model to JSON
with open(MODEL_DIR+model_file, 'w') as json_file:
    json_file.write(model.to_json())


# # 4. Training the CNN

# In[49]:

batch_size = 32
patients_per_epoch = len(train_MRI_set) 
epochs = 50


# In[60]:
version = 'MRI_Gender_Age_FS_'+args.i
#train the model and keep track of progress with history
print('--- Starting training of model ---')

history = LossHistory(epochs, version)
checkpoint=ModelCheckpoint(MODEL_DIR+'model_'+version+'.h5', monitor='c_index', verbose=0, save_best_only=True, mode='max')


steps_per_epoch = int(math.ceil(float(patients_per_epoch)/batch_size))
validation_steps = int(math.ceil(float(validation_size)/batch_size))

model.fit_generator(data_generator(list(train_MRI_set), img_size, batch_size, img_scale, mask, augment=augment_train, mode='train'),
                    steps_per_epoch=steps_per_epoch, #data generator important
                    epochs=epochs,
                    max_queue_size=1,
                    callbacks=[history, checkpoint])#, stoptraining])

print('Succesfully trained the model.')

model.save(MODEL_DIR+'model_'+version+'_last_model.h5')

# In[105]:
#model = tf.keras.models.load_model(MODEL_DIR+'model_'+version+'.h5', custom_objects={'CoxPH_loss': CoxPH_loss})

event_observed = []
event_times = []
for patient in test_MRI_set:
    info = test_label_set.loc[patient]
    event_observed.append(info['dementia'])
    event_times.append(info['event_time'])

test_predictions = model.predict_generator(data_generator(list(test_MRI_set), img_size, batch_size, img_scale, mask, mode='test',shuffle=False),
                                                 steps=int(math.ceil(float(test_size)/batch_size)),
                                                 max_queue_size=1, verbose=2)
output_file = version + '_test_predictions.txt'
with open(MODEL_DIR+'final/'+output_file, "w") as f:
    for element in test_predictions:
        f.write(str(element[0]) + "\n")
f.close()


# In[ ]:

output_file = version + '_test_c_index.txt'
c = 1-concordance_index(event_times,test_predictions, event_observed)
with open(MODEL_DIR+'final/'+output_file, "w") as f:
    f.write('The C-index on the test set is %5.4f' % c)
f.close()

