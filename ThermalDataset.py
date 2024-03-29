import torch 
import pandas as pd

class ThermalDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.features = df[['month', 'day', 'hour', 'minute',
                'room_height', 'room_length', 'room_width', 
                'person_age', 'person_height', 'person_weight', 'person_cloth', 'person_activity',
                'indoor_operative_temperature', 'indoor_mean_radiant_temp', 'indoor_asymmetry_temp', 'indoor_humidity', 'indoor_air_velocity',
                'ashrae_predicted_mean_vote', 'ashrae_predicted_percentage_of_dissatisfied',
                'person_gender_Female', 'person_gender_Male', 
                'season_Summer Season', 'season_Transition Season', 'season_Winter Season',
                'climate_zone_Cold zone',
                'climate_zone_Hot summer and cold winter zone',
                'climate_zone_Severe cold zone', 'building_type _Educational',
                'building_type _Office', 'building_type _Residential',
                'building_function_Bedroom', 'building_function_Classroom',
                'building_function_Living room', 'building_function_Office',
                'building_operation_mode_Air conditioning heating',
                'building_operation_mode_Ceiling capillary heating',
                'building_operation_mode_Cold radiation ceiling cooling',
                'building_operation_mode_Convection cooling',
                'building_operation_mode_Convection heating',
                'building_operation_mode_Naturally Ventilated',
                'building_operation_mode_Others',
                'building_operation_mode_Radiant floor cooling',
                'building_operation_mode_Radiant floor heating',
                'building_operation_mode_Radiator heating']]
        self.thermal_accept = df[["label_thermal_acceptability_vote"]]
        self.thermal_comfort = df[["label_thermal_comfort_vote"]]
        self.thermal_sensation = df[["label_thermal_sensation_vote"]]
        
    def __len__ (self):
        return len(self.df)

    def __getitem__ (self, index:int):
        features = torch.tensor(self.features.loc[index].to_list())
        
        thermal_accept = torch.tensor(self.thermal_accept.loc[index].to_list()[0]).type(torch.LongTensor)
        thermal_comfort = torch.tensor(self.thermal_comfort.loc[index].to_list()[0]).type(torch.LongTensor)
        thermal_sensation = torch.tensor(self.thermal_sensation.loc[index].to_list()[0]).type(torch.LongTensor)
        
        sample = {"features":features, "thermal_accept":thermal_accept, "thermal_comfort":thermal_comfort, "thermal_sensation":thermal_sensation }
        return sample