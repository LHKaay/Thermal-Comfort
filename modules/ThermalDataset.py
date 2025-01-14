import torch 
import pandas as pd

class ThermalDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df = df
        feature_cols = df.columns.to_list()
        feature_cols.remove("D1.TSV")

        self.features = df[feature_cols]
        self.thermal_sensation = df[["D1.TSV"]]
        
    def __len__ (self):
        return len(self.df)

    def __getitem__ (self, index:int):
        features = torch.tensor(self.features.loc[index].to_list())
        
        thermal_sensation = torch.tensor(self.thermal_sensation.loc[index].to_list()).to(torch.float32)
        
        return {"features":features, "thermal_sensation":thermal_sensation }
    

class ThermalDataset_infer(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.features = df
        
    def __len__ (self):
        return len(self.df)

    def __getitem__ (self, index:int):
        features = torch.tensor(self.features.loc[index].to_list())
        
        return {"features":features}