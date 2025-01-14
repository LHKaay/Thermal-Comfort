import torch


class ThermalModel_LossConstruction(torch.nn.Module):
    def __init__(self):
        super(ThermalModel_LossConstruction, self).__init__()
        
        # self.net = torchvision.models.resnet18(pretrained=True)
        self.n_features = 44

        self.fc1 = torch.nn.Linear(self.n_features,self.n_features)
        self.relu1 = torch.nn.ReLU()
        self.final1 = torch.nn.Linear(self.n_features, 4)

        self.fc2 = torch.nn.Linear(self.n_features,self.n_features)
        self.relu2 = torch.nn.ReLU()
        self.final2 = torch.nn.Linear(self.n_features, 5)

        self.fc3 = torch.nn.Linear(self.n_features,self.n_features)
        self.relu3 = torch.nn.ReLU()
        self.final3 = torch.nn.Linear(self.n_features, 7)
        
    def forward(self, x):
        out1 = self.fc1(x)
        out1 = self.relu1(out1)
        thermal_accept = self.final1(out1)

        out2 = self.fc2(x)
        out2 = self.relu2(out2)
        thermal_comfort = self.final2(out2)

        out3 = self.fc3(x)
        out3 = self.relu3(out3)
        thermal_sensation = self.final3(out3)

        return thermal_accept, thermal_comfort, thermal_sensation
    
class ThermalModel_HardParameterSharing(torch.nn.Module):
    def __init__(self):
        super(ThermalModel_HardParameterSharing, self).__init__()
        
        # self.net = torchvision.models.resnet18(pretrained=True)
        self.n_features = 44

        self.inputlayer = torch.nn.Linear(self.n_features,60)
        self.tanh1 = torch.nn.Tanh()
        self.hiddenlayer1 = torch.nn.Linear(60,80)
        self.tanh2 = torch.nn.Tanh()
        self.hiddenlayer2 = torch.nn.Linear(80,100)
        self.tanh3 = torch.nn.Tanh()
        self.hiddenlayer3 = torch.nn.Linear(100,120)
        self.tanh4 = torch.nn.Tanh()
        self.hiddenlayer4 = torch.nn.Linear(120,150)
        self.tanh5 = torch.nn.Tanh()

        self.accept = torch.nn.Linear(150, 1)
        # self.fc2 = torch.nn.Linear(self.n_features,self.n_features)
        # self.relu2 = torch.nn.ReLU()
        self.comfort = torch.nn.Linear(150, 1)
        # self.fc3 = torch.nn.Linear(self.n_features,self.n_features)
        # self.relu3 = torch.nn.ReLU()
        self.sensation = torch.nn.Linear(150, 1)
        
    def forward(self, x):
        out = self.inputlayer(x)

        out = self.tanh1(out)
        out = self.hiddenlayer1(out)

        out = self.tanh2(out)
        out = self.hiddenlayer2(out)

        out = self.tanh3(out)
        out = self.hiddenlayer3(out)

        out = self.tanh4(out)
        out = self.hiddenlayer4(out)

        out = self.tanh5(out)

        thermal_accept = self.accept(out)
        thermal_comfort = self.comfort(out)
        thermal_sensation = self.sensation(out)

        return thermal_accept, thermal_comfort, thermal_sensation
    
class MoELayer(torch.nn.Module):
    '''
    work as the FFN layer 
    '''
    def __init__(self, experts, gate, k=1):
        super(MoELayer, self).__init__()
        assert len(experts) > 0

        self.experts = experts
        self.gate = gate
        self.k = k

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.k)
        weights = torch.nn.functional.softmax(weights,dim=1,dtype=torch.float).type_as(inputs)

        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts==i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
        
        return results.view_as(inputs)

class ThermalModel_MoE(torch.nn.Module):
    '''
    This is the same as Transformer architecture
    '''
    def __init__(self):
        super(ThermalModel_MoE, self).__init__()

        pass

    def forward():

        pass

