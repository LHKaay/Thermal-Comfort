import argparse, yaml, os
import torch
import torchvision
import pandas as pd
import numpy as np
import wandb
import random

from modules import ThermalDataset
from modules import MoE

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import r2_score

device = ("cuda" if torch.cuda.is_available()
          else "cpu")
print(f"Using {device} device !!!")

with open('config/config.yaml', 'r') as f:
    default_config = yaml.load(f, Loader=yaml.SafeLoader)

# Reproducibility
torch.manual_seed(default_config["seed"])
random.seed(default_config["seed"])
np.random.seed(default_config["seed"])

thermal_sensation_map = {-3:"Cold", -2:"Cool", -1:"Slighty cool", 0:"Neutral", 1:"Slighty warm", 2:"Warm", 3:"Hot"}


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

    train = ThermalDataset.ThermalDataset(df_train.reset_index(drop=True))
    valid = ThermalDataset.ThermalDataset(df_valid.reset_index(drop=True))
    test = ThermalDataset.ThermalDataset(df_test.reset_index(drop=True))

    print(f"Training shape: {df_train.shape} \nValidation shape: {df_valid.shape} \nTesting shape: {df_test.shape}")

    return train, valid, test

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def accuracy_cal(targets, prediction):
    prediction = np.squeeze(np.array(prediction))
    targets = np.squeeze(np.array(targets))

    print(f"------------------- Prediction is in range {min(prediction)} ; {max(prediction)} !!!")

    prediction_round = np.clip(np.array([round(y) for y in prediction]),-3,3)
    prediction_lbl = [thermal_sensation_map[y_hat] for y_hat in prediction_round]
    targets_lbl = [thermal_sensation_map[int(target)] for target in targets]
    acc = sum(1 for x,y in zip(prediction_lbl,targets_lbl) if x == y) / len(prediction_lbl)

    return acc

def train(config=default_config):
    run = wandb.init(project=config["project"], entity=config["entity"], name=f"{config['backbone']}, k={config['k']}", job_type="training", config=config)

    if not os.path.exists(f"modelcheckpoints/{config['version']}"):
        os.makedirs(f"modelcheckpoints/{config['version']}")
        os.makedirs(f"modelcheckpoints/{config['version']}/reports")

    LEARNING_RATE = wandb.config.learning_rate
    BATCH_SIZE = wandb.config.batch_size
    EPOCH = wandb.config.epochs

    train, valid, test = read_data(data_path=config["data_path"])

    trainloader = torch.utils.data.DataLoader(train, batch_size= BATCH_SIZE)
    validloader = torch.utils.data.DataLoader(valid, batch_size= BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test, batch_size= BATCH_SIZE)

    model = MoE.MixtureOfExperts(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'], 
                                num_experts=config['num_experts'], backbone=config['backbone'], k=config['k']).to(device=device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss(reduction='mean')
    MAE_loss = torch.nn.L1Loss()
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.00017, steps_per_epoch=len(trainloader), epochs=EPOCH, anneal_strategy='cos')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, )
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(EPOCH):
        #########################################
        ############ Training process ###########
        #########################################
        model.train()
        train_loss = []
        train_mae = []
        y_hats = []
        targets = []
        for idx, data in enumerate(trainloader):
            inputs = data["features"].to(device=device)
            target = data["thermal_sensation"].to(device=device)

            output = model(inputs)
            y_hats.extend(output.cpu().tolist())
            targets.extend(target.cpu().tolist())

            loss = criterion(output, target)
            train_loss.append(loss)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            # scheduler.step()
            mae = MAE_loss(output, target)
            train_mae.append(mae)
            print(f'Step {idx}/{len(trainloader)}, Loss: {loss.item():.4f}\r', end='', flush=True)
            # wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})
            wandb.log({"Learning Rate": get_lr(optimizer)})

        train_acc = accuracy_cal(targets=targets, prediction=y_hats)

        #########################################
        ########## Validation process ###########
        #########################################
        model.eval()
        valid_loss = []
        valid_mae_loss = []
        y_hats = []
        targets = []

        with torch.no_grad():
            for idx, data in enumerate(validloader):
                inputs = data["features"].to(device=device)
                target = data["thermal_sensation"].to(device=device)

                output = model(inputs)
                y_hats.extend(output.cpu().tolist())
                targets.extend(target.cpu().tolist())

                loss = criterion(output, target)
                mae = MAE_loss(output, target)

                valid_loss.append(loss.item())
                valid_mae_loss.append(mae.item())
        val_acc = accuracy_cal(targets=targets, prediction=y_hats)
        valid_loss = sum(valid_loss)/len(valid_loss)
        valid_mae = sum(valid_mae_loss)/len(valid_mae_loss)
        train_loss = sum(train_loss)/len(train_loss)
        train_mae = sum(train_mae)/len(train_mae)

        print(f"Epoch {epoch+1} / {EPOCH}, Train Loss = {train_loss:.4f}, Train MAE = {train_mae:.4f}, Train Acc = {train_acc:.4f}, Valid Loss:  {valid_loss:.4f}, Valid MAE Loss:  {valid_mae:.4f}, Valid Acc: {val_acc:.4f}")

        wandb.log({ "Train MSE Loss": train_loss, "Train MAE Loss": train_mae, "Valid MSE Loss": valid_loss, "Valid MAE Loss": valid_mae, "Train Acc": train_acc, "Valid Acc": val_acc})
        scheduler.step()
        
    torch.save(model.state_dict(), f'modelcheckpoints/{config["version"]}/model_v{default_config["version"]}_{default_config["backbone"]}.pth')
    wandb.save(f'modelcheckpoints/{config["version"]}/model_v{default_config["version"]}_{default_config["backbone"]}.pth')

    #########################################
    ############ Testing process ############
    #########################################
    model.eval()
    test_loss = []
    criterion = torch.nn.MSELoss()

    y_hats = []
    targets = []

    with torch.no_grad():
        for idx, data in enumerate(testloader):
            inputs = data["features"].to(device=device)
            target = data["thermal_sensation"].to(device=device)

            output = model(inputs)

            y_hats.extend(output.cpu().tolist())
            targets.extend(target.cpu().tolist())

            loss = criterion(output, target)

            test_loss.append(loss.item())

    y_hats = np.squeeze(np.array(y_hats))
    targets = np.squeeze(np.array(targets))

    test_loss = sum(test_loss)/ len(test_loss)

    y_hats_round = np.clip(np.array([round(y_hat) for y_hat in y_hats]),-3,3)
    y_hats_lbl = [thermal_sensation_map[y_hat] for y_hat in y_hats_round]
    targets_lbl = [thermal_sensation_map[int(target)] for target in targets]
    test_acc = sum(1 for x,y in zip(y_hats_lbl,targets_lbl) if x == y) / len(y_hats_lbl)
    test_r2 = r2_score(y_true=targets, y_pred=y_hats_round)
    result_df = pd.DataFrame({
        "Target":targets,
        "Y_hat":y_hats_round,
        "Target_Sensation":targets_lbl,
        "Y_hat_Sensation": y_hats_lbl
    })
    result_df.to_csv(f'modelcheckpoints/{config["version"]}/reports/{config["version"]}_{default_config["backbone"]}.csv')

    fig = make_subplots()

    fig.add_trace(
        go.Scatter(x=result_df.index, y=result_df.Target, text=result_df.Target_Sensation, name="Ground Truth"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=result_df.index, y=result_df.Y_hat, text=result_df.Y_hat_Sensation, name="Prediction"),
        secondary_y=False
    )

    fig.update_layout(title_text=f"{config['version']} with {test_acc:.4f} acc", hovermode="x")
    fig.update_xaxes(title_text="Index")
    fig.update_yaxes(title_text="Thermal Sensation Value", secondary_y=False)

    fig.write_html(f'modelcheckpoints/{config["version"]}/reports/{default_config["version"]}_{default_config["backbone"]}.html')
    
    print(f"Train Loss = {train_loss:.4f}, Valid Loss:  {valid_loss:.4f}, Test Loss:  {test_loss:.4f}, Test acc: {test_acc:.4f}, Test r2: {test_r2:.4f}")

    run.finish()
    return model

if __name__=='__main__':
    parse_args()
    print(default_config)
    model = train(default_config)
