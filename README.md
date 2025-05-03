# DFT-Prediction-ANN

This model was trained on a subset of data from the ANI GDB dataset. It consists of 864,898 molecules containing the four atoms, H, C, N, and O, which range in size from 2 to 14 atoms in total. It also can not be uploaded here because it is too large.

After training, it was able to predict DFT energies well, to within 2 kCal/mol Root Mean Squared Error. 

## Methods / Functions
### init_aev_computer()
This function initiates the Atomic Environment Vector (AEV) computer. This computer is essential for processing the inputs of the dataset so that it can be passed into the model.

### load_ani_dataset(path)
Loads the ANI dataset 

### ANITrainer()
A class to trainer the model
#### init(model, batch_size, learning_rate, epoch, l2))
Initializes the trainer, setting hyperparameters, and initializing the optimizer. This model uses ADAM optimization.
#### train(train_data, val_data, early_stop=True, draw_curve=True, train_len=-1)
Trains the model. Calculates loss of predicted energy with a mean squared error loss function, calculates gradient of the model's parameters, and adjusts them. 
#### evaluate(data, draw_plot=False, batch_size=1024)
Evaluates loss of the function on the input data.

### AtomicNet()
The model's ANN
#### init
Creates the ANN, this ANN uses CELU activation functions with a parameter of 0.1, along with a dropout after the first two layers with a setting of 0.1. There are four Linear layers total.

### KFoldCrossValidation(model, k, training_data, testing_data, opt_method='adam', learning_rate=2e-3, batch_size=128, epoch=50, l2=0.0)
Trains the model k times against k rearragements of the training data. In the end, this function outputs the average testing and training losses.
