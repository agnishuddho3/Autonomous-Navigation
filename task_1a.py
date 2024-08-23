'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_3179
# Author List:		Aditya Raj, Aditya Ravindra Jadhav, Agnishuddho Kundu, Aryansh Kumar
# Filename:			task_1a.py
# Functions:	    [ideantify_features_and_targets, load_as_tensors,
# 					 model_loss_function, model_optimizer, model_number_of_epochs, training_function,
# 					 validation_functions ]

####################### IMPORT MODULES #######################
import pandas
import torch
import numpy as np
###################### Additional Imports ####################
'''
You can import any additional modules that you require from
torch, matplotlib or sklearn.
You are NOT allowed to import any other libraries. It will
cause errors while running the executable

'''

import torch.nn as nn


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx]).float(), (self.y[idx]).float()



##############################################################

def data_preprocessing(task_1a_dataframe):
    '''

	 	Purpose:
	---
	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that
	there are features in the csv file whose values are textual (eg: Industry,
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function
	should return encoded dataframe in which all the textual features are
	numerically labeled.

	Input Arguments:
	---
	task_1a_dataframe: [Dataframe]
						  Pandas dataframe read from the provided dataset

	Returns:
	---
	encoded_dataframe : [ Dataframe ]
						  Pandas dataframe that has all the features mapped to
						  numbers starting from zero

	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''


	#################	ADD YOUR CODE HERE	##################
    encoded_dataframe = task_1a_dataframe
    le = LabelEncoder()
    encoded_dataframe['Gender'] = le.fit_transform(encoded_dataframe['Gender'])
    encoded_dataframe['EverBenched'] = le.fit_transform(encoded_dataframe['EverBenched'])
    encoded_dataframe['Education'] =  le.fit_transform(encoded_dataframe['Education'])
    encoded_dataframe['City'] =  le.fit_transform(encoded_dataframe['City'])
    encoded_dataframe['JoiningYear'] =  le.fit_transform(encoded_dataframe['JoiningYear'])


	##########################################################

    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
	'''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second
	item is the target label

	Input Arguments:
	---
	encoded_dataframe : [ Dataframe ]
						Pandas dataframe that has all the features mapped to
						numbers starting from zero

	Returns:
	---
	features_and_targets : [ list ]
							python list in which the first item is the
							selected features and second item is the target label

	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################
	features_and_targets=encoded_dataframe
	##########################################################

	return features_and_targets


def load_as_tensors(features_and_targets):

	#################	ADD YOUR CODE HERE	##################
    X_train, X_test, y_train, y_test =train_test_split(features_and_targets.iloc[:,:-1], features_and_targets.iloc[:,[-1]], test_size = 0.2, random_state = 42)
    sc = StandardScaler()

    X_train.iloc[:,[3,4] ] = sc.fit_transform(X_train.iloc[:,[3,4]])
    X_test.iloc[:,[3,4]] = sc.transform(X_test.iloc[:,[3,4]])

    X_train = X_train.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)


    X_train = X_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    X_test = X_test.to(torch.float32)
    y_test = y_test.to(torch.float32)



    train_loader_X =  DataLoader(dataset=X_train ,batch_size = 32 ,shuffle =True , num_workers=2)    ## num workers 0 means that all the data will be loaded in the main proces
    train_loader_y =  DataLoader(dataset=y_train ,batch_size = 32 ,shuffle =True , num_workers=2)    ## num workers 0 means that all the data will be loaded in the main proces
    test_loader_X=  DataLoader(dataset=X_test ,batch_size = 32 ,shuffle =True , num_workers=2)    ## num workers 0 means that all the data will be loaded in the main proces
    test_loader_y=  DataLoader(dataset=y_test ,batch_size = 32 ,shuffle =True , num_workers=2)    ## num workers 0 means that all the data will be loaded in the main proces

    ## creating iterables
    train_iter_X = iter(train_loader_X)
    train_iter_y = iter(train_loader_y)
    test_iter_y = iter(test_loader_y)
    test_iter_X = iter(test_loader_X)

    ## storing iterables in a list
    # iter_lst = [train_loader_X,test_loader_X,train_loader_y,test_loader_y]

    tensors_and_iterable_training_data = [X_train , X_test , y_train , y_test]
	##########################################################init
    return tensors_and_iterable_training_data


class Salary_Predictor(nn.Module):
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        self.l1 = nn.Linear(8, 128)
        torch.nn.init.kaiming_uniform_(self.l1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.l2 = nn.Linear(128, 128)
        torch.nn.init.kaiming_uniform_(self.l2.weight, nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(128)


        self.l3 = nn.Linear(128, 32)
        torch.nn.init.kaiming_uniform_(self.l3.weight, nonlinearity='relu')
        self.bn3 = nn.BatchNorm1d(32)
        self.l4 = nn.Linear(32, 1)
        torch.nn.init.kaiming_uniform_(self.l4.weight, nonlinearity='linear')


    def forward(self, x):

        # if x.dim() == 1:
        #     x = x.unsqueeze(0)

        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.l2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.l3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.l4(out)
        # out = self.sigmoid(out)


        predicted_output = out
        return predicted_output


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):

    for epoch in range(model_number_of_epochs()):
      X_train = tensors_and_iterable_training_data[0]
      y_train = tensors_and_iterable_training_data[2]
      train_dataset = CustomDataset(X_train,y_train)
      train_loader = DataLoader(dataset=train_dataset, batch_size=32,shuffle=True, num_workers=0)


      for i,(X_data,y_data) in enumerate(train_loader):


        optimizer.zero_grad()
        y_pred = model(X_data)
        y_data=y_data.reshape(-1,1)
        ls = loss_function(y_pred, y_data)
        ls.backward()
        optimizer.step()

        # if (i + 1) % 10 == 0:
        #         print(f'Epoch [{epoch + 1}/{number_of_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {ls.item():.4f}')

    trained_model = model
    return trained_model
def model_loss_function():

	#################	ADD YOUR CODE HERE	##################
    loss_function =nn.BCEWithLogitsLoss()

	##########################################################
    return loss_function

def model_optimizer(model):

	#################	ADD YOUR CODE HERE	##################
	optimizer = torch.optim.Adamax(model.parameters(), lr =0.01)

	##########################################################

	return optimizer

def model_number_of_epochs():

	#################	ADD YOUR CODE HERE	##################
    number_of_epochs = 100
	##########################################################
    return number_of_epochs

def validation_function(trained_model, tensors_and_iterable_training_data):


    n_correct = 0
    n_samples = 0
    X_test = tensors_and_iterable_training_data[1]
    y_test = tensors_and_iterable_training_data[3]
    test_dataset = CustomDataset(X_test,y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32,shuffle=True, num_workers=0)
    for (X_data,y_data) in test_loader:

        output = trained_model(X_data)
        output = (output>=0.0).to(torch.float32)
        n_samples += y_data.shape[0]
        n_correct += (output == y_data).sum().item()
        model_accuracy = 100*(n_correct/n_samples)

    return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":

  task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')


  encoded_dataframe = data_preprocessing(task_1a_dataframe)
  encoded_dataframe

  features_and_targets = identify_features_and_targets(encoded_dataframe)


  tensors_and_iterable_training_data = load_as_tensors(features_and_targets)


  model = Salary_Predictor()


  loss_function = model_loss_function()
  optimizer = model_optimizer(model)
  number_of_epochs = model_number_of_epochs()

  trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data,
					loss_function, optimizer)

  model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
  print(f"Accuracy on the test set = {model_accuracy}")


  X_train = tensors_and_iterable_training_data[0]
  train_loader_X =  DataLoader(dataset=X_train ,batch_size = 32 ,shuffle =True , num_workers=2)    ## num workers 0 means that all the data will be loaded in the main proces

  train_iter_X = iter(train_loader_X)
  X_train_tensor = next(train_iter_X).unsqueeze(0)


  x = X_train_tensor[0]

  jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
