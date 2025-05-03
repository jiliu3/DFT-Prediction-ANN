import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchani
import torchani.data
import itertools
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from itertools import chain
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import h5py
with h5py.File('./ani_gdb_s01_to_s04.h5', 'r') as f:
    print(list(f.keys()))

#create a atomic environment vector computer to process inputs
def init_aev_computer():
    Rcr = 5.2
    Rca = 3.5
    EtaR = torch.tensor([16], dtype=torch.float, device=device)
    ShfR = torch.tensor([
        0.900000, 1.168750, 1.437500, 1.706250, 
        1.975000, 2.243750, 2.512500, 2.781250, 
        3.050000, 3.318750, 3.587500, 3.856250, 
        4.125000, 4.393750, 4.662500, 4.931250
    ], dtype=torch.float, device=device)


    EtaA = torch.tensor([8], dtype=torch.float, device=device)
    Zeta = torch.tensor([32], dtype=torch.float, device=device)
    ShfA = torch.tensor([0.90, 1.55, 2.20, 2.85], dtype=torch.float, device=device)
    ShfZ = torch.tensor([
        0.19634954, 0.58904862, 0.9817477, 1.37444680, 
        1.76714590, 2.15984490, 2.5525440, 2.94524300
    ], dtype=torch.float, device=device)

    num_species = 4
    aev_computer = torchani.AEVComputer(
        Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species
    )
    return aev_computer

aev_computer = init_aev_computer()
aev_dim = aev_computer.aev_length
print(aev_dim)

#load dataset
def load_ani_dataset(dspath):
    self_energies = torch.tensor([
        0.500607632585, -37.8302333826,
        -54.5680045287, -75.0362229210
    ], dtype=torch.float, device=device)
    energy_shifter = torchani.utils.EnergyShifter(None)
    species_order = ['H', 'C', 'N', 'O']

    dataset = torchani.data.load(dspath)
    dataset = dataset.subtract_self_energies(energy_shifter, species_order)
    dataset = dataset.species_to_indices(species_order)
    dataset = dataset.shuffle()
    return dataset

dataset = load_ani_dataset("./ani_gdb_s01_to_s04.h5")
# Use dataset.split method to do split
train_data, val_data, test_data = dataset.split(0.6, 0.2, None)

#Create the trainer class
class ANITrainer:
    def __init__(self, model, batch_size, learning_rate, epoch, l2):
        self.model = model
        
        num_params = sum(item.numel() for item in model.parameters())
        print(f"{model.__class__.__name__} - Number of parameters: {num_params}")
        
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=l2)
        self.epoch = epoch
    
    def train(self, train_data, val_data, early_stop=True, draw_curve=True, train_len=-1):
        self.model.train()
        
        # init data loader
        print("Initialize training data...")
        #train_data_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        train_data_loader = train_data.collate(self.batch_size).cache()
        val_data_loader = val_data.collate(self.batch_size).cache()
        
        # definition of loss function: MSE is a good choice! 
        loss_func = nn.MSELoss()
        
        # record epoch losses
        train_loss_list = []
        val_loss_list = []
        lowest_val_loss = np.inf

        if train_len == -1:
            train_len = len(train_data)
        
        for i in tqdm(range(self.epoch), leave=True):
            train_epoch_loss = 0.0
            for train_data_batch in train_data_loader:

                species = train_data_batch['species'].to(device)
                coords = train_data_batch['coordinates'].to(device)
                true_energies = train_data_batch['energies'].to(device).float()
                _, pred_energies = model((species, coords))
                
                # compute energies
                #batch_importance = train_data_batch.shape[0] / train_data
                
                # compute loss
                batch_loss = loss_func(true_energies, pred_energies)
                
                # do a step
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                batch_importance = len(train_data_batch) / train_len
                train_epoch_loss += batch_loss * batch_importance

            #train_loss_list.append(train_epoch_loss.detach().numpy())
            train_loss_list.append(train_epoch_loss.detach().cpu().numpy())

            
            # use the self.evaluate to get loss on the validation set 
            
            val_epoch_loss, _, _ = self.evaluate(val_data)
            
            # append the losses
            val_loss_list.append(val_epoch_loss)
            
            if early_stop:
                if val_epoch_loss < lowest_val_loss:
                    lowest_val_loss = val_epoch_loss
                    weights = self.model.state_dict()
        
        if draw_curve:
            x_axis = list(range(1, len(train_loss_list) + 1))
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
            ax.set_yscale("log")
            # Plot train loss and validation loss
            ax.plot(x_axis, train_loss_list, label='Train')
            ax.plot(x_axis, val_loss_list, label='Validation')
            ax.legend()
            ax.set_xlabel("# Batch")
            ax.set_ylabel("Loss")
        
        if early_stop:
            self.model.load_state_dict(weights)
        
        return train_loss_list, val_loss_list
    
    
    def evaluate(self, data, draw_plot=False, batch_size=1024):
        self.model.eval()
        
        # init loss function
        loss_func = nn.MSELoss()
        total_loss = 0.0

        data_loader = data.collate(batch_size).cache()
        
        true_energies_all = []
        pred_energies_all = []

        with torch.no_grad():
            for batch_data in data_loader:
                
                # compute energies
                species = batch_data['species'].to(device)
                coords = batch_data['coordinates'].to(device)
                true_energies = batch_data['energies'].to(device).float()
                _, pred_energies = self.model((species, coords))
                
                # compute loss
                batch_loss = loss_func(pred_energies, true_energies)

                batch_importance = len(batch_data) / len(data)
                total_loss += batch_importance * batch_loss.item()
                
                
                true_energies_all.append(true_energies.detach().cpu().numpy().flatten())
                pred_energies_all.append(pred_energies.detach().cpu().numpy().flatten())
            true_energies_all = np.concatenate(true_energies_all)
            pred_energies_all = np.concatenate(pred_energies_all)
            hartree2kcalmol = 627.5094738898777
            mae =  np.mean(np.abs(true_energies_all - pred_energies_all)) * hartree2kcalmol
            rmse = np.sqrt(np.mean((true_energies_all - pred_energies_all)**2)) * hartree2kcalmol

        if draw_plot:
            # Report the mean absolute error
            # The unit of energies in the dataset is hartree
            # please convert it to kcal/mol when reporting the mean absolute error
            # 1 hartree = 627.5094738898777 kcal/mol
            # MAE = mean(|true - pred|)
            #print(f"mae {mae}")
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
            ax.scatter(true_energies_all, pred_energies_all, label=f"MAE: {mae:.2f} kcal/mol\nRMSE: {rmse:.2f} kcal/mol", s=2)
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Predicted")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            vmin, vmax = min(xmin, ymin), max(xmax, ymax)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.plot([vmin, vmax], [vmin, vmax], color='red')
            ax.legend()
            
        return total_loss, mae, rmse

#Create the model to use
class AtomicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 200),
            nn.CELU(0.1),
            nn.Dropout(0.1),
            nn.Linear(200, 120),
            nn.CELU(0.1),
            nn.Dropout(0.1),
            nn.Linear(120, 90),
            nn.CELU(0.1),
            nn.Linear(90, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

net_H = AtomicNet()
net_C = AtomicNet()
net_N = AtomicNet()
net_O = AtomicNet()

aev_computer = init_aev_computer()
aev_dim = aev_computer.aev_length

# ANI model requires a network for each atom type
# use torch.ANIModel() to compile atomic networks
ani_net = torchani.ANIModel([net_H, net_C, net_N, net_O])
model = nn.Sequential(
    aev_computer,
    ani_net
).to(device)


#Implement K-Fold Cross Validation
def KFoldCrossValidation(
    model, k, 
    training_data, testing_data, 
    opt_method='adam', learning_rate=2e-3, batch_size=128, epoch=50, l2=0.0
):

    kf = KFold(n_splits=k)

    splits = training_data.split(*np.full(k, 1/k))

    print(len(training_data))
    print(len(splits[0]))
    
    train_acc_list, test_acc_list = [], []
    mae_list, rmse_list = [], []
    
    for i in range(len(splits)):

        model_ = new_model()

        train_len = len(training_data) - len(splits[i])
        
        print(f"Fold {i}:")

        train_data = torchani.data.TransformableIterable(chain(*(splits[0:i] + splits[i+1:])))
        val_data = splits[i]
        

        # initialize a Trainer object
        trainer = ANITrainer(model_, batch_size=batch_size, learning_rate=learning_rate, epoch=epoch, l2=l2)
        # call trainer.train() here
        res = trainer.train(train_data, val_data,train_len=train_len)
        #print(res)
        train_acc_best = res[0][np.argmin(res[1])]
        test_loss, mae, rmse = trainer.evaluate(testing_data, draw_plot=True)
        
        train_acc_list.append(train_acc_best)
        test_acc_list.append(test_loss)
        mae_list.append(mae)
        rmse_list.append(rmse)
        
        print(f"Training loss: {train_acc_best}")
        print(f"Testing loss: {test_loss}")
        print(f"Testing mae: {mae} rmse: {rmse}")
    
    print("Final results:")
    print(f"Training loss: {np.mean(train_acc_list)}+/-{np.std(train_acc_list)}")
    print(f"Testing loss: {np.mean(test_acc_list)}+/-{np.std(test_acc_list)}")
    print(f"MAE: {np.mean(mae_list)}+/-{np.std(mae_list)}")
    print(f"RMSE: {np.mean(rmse_list)}+/-{np.std(rmse_list)}")

#Run the model
KFoldCrossValidation(model, 3, train_data, test_data, batch_size=8192, learning_rate=2e-5, epoch=400, l2=1e-6)
