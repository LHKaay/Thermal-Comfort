import argparse, yaml, os
import torch
import torchvision
import pandas as pd
import numpy as np
import wandb
import random

from ThermalDataset import ThermalDataset
from ThermalModel import ThermalModel_HardParameterSharing


device = ("cuda" if torch.cuda.is_available()
          else "cpu")
print(f"Using {device} device !!!")

with open('config.yaml', 'r') as f:
    default_config = yaml.load(f, Loader=yaml.SafeLoader)

# Ensure that results are reproducible
random.seed(42)

def parse_args():
    "Override default argments"
    argparser = argparse.ArgumentParser(description="Hyper-parameter")
    argparser.add_argument('--batch_size', type=int, default=default_config['batch_size'], help='Batch size')
    argparser.add_argument('--learning_rate', type=float, default=default_config["learning_rate"], help='learning rate')
    argparser.add_argument('--epochs', type=int, default=default_config["epochs"], help='number of training epochs')

    args = vars(argparser.parse_args())
    default_config.update(args)
    return 

def read_data(data_path):
    '''
    Read the training data
    '''

    df = pd.read_csv(data_path)
    df_train, df_valid, df_test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])

    train = ThermalDataset(df_train.reset_index(drop=True))
    valid = ThermalDataset(df_valid.reset_index(drop=True))
    test = ThermalDataset(df_test.reset_index(drop=True))

    print(f"Training shape: {df_train.shape} \nValidation shpae: {df_valid.shape} \nTesting shape: {df_test.shape}")

    return train, valid, test

def train(config=default_config):
    run = wandb.init(project=config["project"], entity=config["entity"], job_type="training", config=config)
    
    LEARNING_RATE = wandb.config.learning_rate
    BATCH_SIZE = wandb.config.batch_size
    EPOCH = wandb.config.epochs

    train, valid, test = read_data(data_path=config["data_path"])

    model = ThermalModel_HardParameterSharing().to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.09)

    ts_loss = torch.nn.CrossEntropyLoss()
    tc_loss = torch.nn.CrossEntropyLoss()
    ta_loss = torch.nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(train, batch_size= BATCH_SIZE)
    validloader = torch.utils.data.DataLoader(valid, batch_size= BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test, batch_size= BATCH_SIZE)

    for epoch in range(EPOCH):
        model.train()
        #########################################
        ############ Training process ###########
        #########################################
        training_accept_loss = 0.0
        training_comfort_loss = 0.0
        training_sensation_loss = 0.0

        training_accept_correct = 0
        training_comfort_correct = 0
        training_sensation_correct = 0
        for i, data in enumerate(trainloader):
            inputs = data["features"].to(device=device)

            thermal_accept = data["thermal_accept"].to(device=device)
            thermal_comfort = data["thermal_comfort"].to(device=device)
            thermal_sensation = data["thermal_sensation"].to(device=device)

            thermal_accept_output, thermal_comfort_output, thermal_sensation_output  = model(inputs)

            loss_ta = ta_loss(thermal_accept_output, thermal_accept)
            loss_tc = tc_loss(thermal_comfort_output, thermal_comfort)
            loss_ts = ts_loss(thermal_sensation_output, thermal_sensation)

            optimizer.zero_grad()

            total_loss =  loss_ta + loss_tc + loss_ts
            total_loss.backward()
            optimizer.step()

            training_accept_loss += loss_ta.item()
            training_comfort_loss += loss_tc.item()
            training_sensation_loss += loss_ts.item()

            _, thermal_accept_predicted = torch.max(thermal_accept_output,1)
            training_accept_correct += (thermal_accept_predicted == thermal_accept).sum().item()

            _, thermal_comfort_predicted = torch.max(thermal_comfort_output,1)
            training_comfort_correct += (thermal_comfort_predicted == thermal_comfort).sum().item()

            _, thermal_sensation_predicted = torch.max(thermal_sensation_output,1)
            training_sensation_correct += (thermal_sensation_predicted == thermal_sensation).sum().item()
        
        #########################################
        ########## Validation process ###########
        #########################################
        valid_accept_loss = 0.0
        valid_comfort_loss = 0.0
        valid_sensation_loss = 0.0

        valid_accept_correct = 0
        valid_comfort_correct = 0
        valid_sensation_correct = 0
        # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validloader):
                inputs = vdata["features"].to(device=device)

                thermal_accept = vdata["thermal_accept"].to(device=device)
                thermal_comfort = vdata["thermal_comfort"].to(device=device)
                thermal_sensation = vdata["thermal_sensation"].to(device=device)

                thermal_accept_output, thermal_comfort_output, thermal_sensation_output  = model(inputs)

                vloss_ta = ta_loss(thermal_accept_output, thermal_accept)
                vloss_tc = tc_loss(thermal_comfort_output, thermal_comfort)
                vloss_ts = ts_loss(thermal_sensation_output, thermal_sensation)

                total_vloss =  vloss_ta + vloss_tc + vloss_ts

                valid_accept_loss += vloss_ta.item()
                valid_comfort_loss += vloss_tc.item()
                valid_sensation_loss += vloss_ts.item()

                _, thermal_accept_predicted = torch.max(thermal_accept_output,1)
                valid_accept_correct += (thermal_accept_predicted == thermal_accept).sum().item()

                _, thermal_comfort_predicted = torch.max(thermal_comfort_output,1)
                valid_comfort_correct += (thermal_comfort_predicted == thermal_comfort).sum().item()

                _, thermal_sensation_predicted = torch.max(thermal_sensation_output,1)
                valid_sensation_correct += (thermal_sensation_predicted == thermal_sensation).sum().item()
        
        print(f"epoch {epoch+1} / {EPOCH}, loss = {total_loss.item():.4f}, valid loss:  {total_vloss.item():.4f}")

        wandb.log({"Thermal Accept Loss": training_accept_loss/len(train), 
                    "Thermal Comfort Loss": training_comfort_loss/len(train), 
                    "Thermal Sensation Loss": training_sensation_loss/len(train),

                    "Thermal Accept Accuracy": training_accept_correct/len(train), 
                    "Thermal Comfort Accuracy": training_comfort_correct/len(train), 
                    "Thermal Sensation Accuracy": training_sensation_correct/len(train),


                    "Thermal Accept Valid Loss": valid_accept_loss/len(valid), 
                    "Thermal Comfort Valid Loss": valid_comfort_loss/len(valid), 
                    "Thermal Sensation Valid Loss": valid_sensation_loss/len(valid),

                    "Total_Valid_Loss": total_vloss,

                    "Thermal Accept Valid Accuracy": valid_accept_correct/len(valid), 
                    "Thermal Comfort Valid Accuracy": valid_comfort_correct/len(valid), 
                    "Thermal Sensation Valid Accuracy": valid_sensation_correct/len(valid)})

if __name__=='__main__':
    parse_args()
    print(default_config)
    train(default_config)
