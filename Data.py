# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:07:51 2020

@author: The Jipsess
"""

# Import required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% Loading & pre-processing of the data
def init(feature = 'DILI1', file = 'p7-mcf7-camda2020.csv', return_all = False,
         upsample = False, downsample = False):
    if downsample == True & upsample == True:
        raise Exception("Downsample and upsample cannot both be True!")
    
    # Load the requested input data  
    if file == 'all':
        file = ['p3-phh-camda2020.csv',
                'p4-hepg2-camda2020.csv',
                'p5-haie-camda2020.csv',
                'p6-a375-camda2020.csv',
                'p7-mcf7-camda2020.csv',
                'p8-pc3-camda2020.csv']
    if type(file) == str:
        # Load one input file
        X = pd.read_csv(f'Data/{file}')
    elif type(file) == list:
        # Load all requested inputs and merge into one dataframe
        X = pd.read_csv(f'Data/{file[0]}')
        X = X.add_suffix(f'_{file[0][0:2]}')
        X = X.rename(columns = {f'CAM_ID_{file[0][0:2]}' : 'CAM_ID'})
        for i in range(1, len(file)):
            X = pd.merge(X,
                         pd.read_csv(f'Data/{file[i]}').add_suffix(f'_{file[i][0:2]}'),
                         left_on = 'CAM_ID',
                         right_on = f'CAM_ID_{file[i][0:2]}',
                         validate = 'one_to_one',
                         sort = False)
            # Remove double sample column
            X.drop(f'CAM_ID_{file[i][0:2]}', axis = 1, inplace = True)
    
    # Load the labels of the data
    Y = pd.read_csv('Data/targets-camda2020.csv')
    
    # Only keep samples for which training data is available.
    Y = Y.loc[Y.CAM_ID.isin(X.CAM_ID)]
    
    # Create identical row order for X & Y
    X = X.sort_values('CAM_ID').reset_index(drop = True)
    Y = Y.sort_values('CAM_ID').reset_index(drop = True)
    
    # Ensure that X and Y have identical CAM_IDs in each row
    if not Y.CAM_ID.equals(X.CAM_ID):
        raise Exception('X != Y...  make sure the rows of X and Y describe the ' +
                        'same samples!!')
    
    
    # Seperate the validation sets
    Y_val = Y.loc[Y.Training_Validation != 'Training Set']
    X_val = X.loc[X.CAM_ID.isin(Y_val.CAM_ID)]
    
    # Remove validation sets from the test and training sets
    Y = Y.loc[Y.Training_Validation == 'Training Set']
    X = X.loc[X.CAM_ID.isin(Y.CAM_ID)]
    
    if upsample:
        X.drop('CAM_ID',axis = 1, inplace = True)
        Y.drop('CAM_ID',axis = 1, inplace = True)
    else:
        # Turn ID column into rownames
        X.set_index('CAM_ID', inplace = True, verify_integrity = True)
        Y.set_index('CAM_ID', inplace = True, verify_integrity = True)
    
    # Create the test and training set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.2,
                                                    random_state =
                                                    np.random.randint(0, 10000))
    
    
    if upsample == True:
        #  Up-sample the minority class in the data to balance the classes
        from sklearn.utils import resample
        
        df = pd.concat([X_train, Y_train], axis = 1)
        
        # Separate majority and minority classes
        df_majority = df[df.DILI1==0]
        df_minority = df[df.DILI1==1]
         
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                          replace=True,                      # sample with replacement
                                          n_samples=df_majority.shape[0],    # to match majority class
                                          random_state=1)                    # reproducible results
         
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
         
        # Shuffle the rows such that the dataframe is not sorted on class
        df_upsampled = df_upsampled.sample(frac=1)
        
        # Display new class counts
        df_upsampled.DILI1.value_counts()
        
        X_train = df_upsampled.iloc[:,0:df_upsampled.shape[1]-7]
        Y_train = df_upsampled.iloc[:,df_upsampled.shape[1]-7:df_upsampled.shape[1]]
        
    if downsample == True:
        #  Up-sample the minority class in the data to balance the classes
        from sklearn.utils import resample
        
        df = pd.concat([X_train, Y_train], axis = 1)
        
        # Separate majority and minority classes
        df_majority = df[df.DILI1==0]
        df_minority = df[df.DILI1==1]
         
        # Upsample minority class
        df_majority_downsampled = resample(df_majority, 
                                           replace=True,
                                           n_samples=df_minority.shape[0],
                                           random_state=1)        
         
        # Combine majority class with upsampled minority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        
        # Shuffle the rows such that the dataframe is not sorted on class
        df_downsampled = df_downsampled.sample(frac=1)
        
        # Display new class counts
        df_downsampled.DILI1.value_counts()
        
        X_train = df_downsampled.iloc[:,0:df_upsampled.shape[1]-7]
        Y_train = df_downsampled.iloc[:,df_upsampled.shape[1]-7:df_upsampled.shape[1]]
    
 
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Set feature to train/test the model on
    Y_train_all = Y_train
    Y_train = Y_train[feature]
    Y_test_all = Y_test
    Y_test = Y_test[feature]
    
    if return_all:
        return(X_train, Y_train, X_test, Y_test, Y_train_all, Y_test_all)
    else:
        return(X_train, Y_train, X_test, Y_test)
    
    
    def feature_filter(data, feat_index):
        # This function takes your X_train & X_test data and filters out the
        # selected features based on the feat_index which is a boolean indexer.
        for i in range(len(data)):
            
            # Filter appropriately depending on data type
            if type(data[i]) == np.ndarray:
                data[i] = data[i][:, feat_index]
            else:
                datac[i] = data[i].iloc[:, feat_index]
                
        return data