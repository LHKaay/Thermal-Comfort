import sys, yaml
sys.path.append('../Thermal-Comfort')

import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from modules import MoE, ThermalDataset


thermal_sensation_map = {-3:"Cold", -2:"Cool", -1:"Slighty cool", 0:"Neutral", 1:"Slighty warm", 2:"Warm", 3:"Hot"}

with open('config/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device !!!")

chinese_dataset = pd.read_csv("dataset/clean_data/chinese_dataset_before_scale.csv")
scaler = preprocessing.MinMaxScaler()
feature_cols = chinese_dataset.columns.to_list()
feature_cols.remove('D1.TSV')
scaler.fit(chinese_dataset[feature_cols])

model = MoE.MixtureOfExperts(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'], 
                                num_experts=config['num_experts'], backbone=config['backbone'], k=config['k']).to(device=device)
weights = torch.load(f'modelcheckpoints/{config["version"]}/model_v{config["version"]}_{config["backbone"]}.pth')
model.load_state_dict(weights)
model.eval()

def model_pipeline(input_data: list):
    input_data = pd.DataFrame([input_data])
    input_data = scaler.transform(input_data)
    input_data = pd.DataFrame(input_data)

    features = ThermalDataset.ThermalDataset_infer(input_data.reset_index(drop=True))

    with torch.no_grad():
        inputs = torch.unsqueeze(features[0]["features"], dim=0).to(device=device)
        thermal_sensation_output  = model(inputs)

    thermal_sensation_output = np.array([[thermal_sensation_output.item()]])
    thermal_sensation_output = thermal_sensation_map[round(thermal_sensation_output)]
    return  thermal_sensation_output

if __name__=='__main__':
    input_data = [163.0, 48.0, 1.05, 1.0, 0.51, 10.49, 24.8, 49.0, 0.07, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    thermal_sensation_value = model_pipeline(input_data)

    print(thermal_sensation_value)
    
