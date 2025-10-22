import numpy as np 
import torch 
from config import Config
import pandas as pd 


class SimulationDataset_wo_1():
    def __init__(self):
        self.csv_path = "datasets/simulation_gene_data.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 2.00, 3.00, 4.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t][["x1", "x2"]].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    

class SimulationDataset_wo_2():
    def __init__(self):
        self.csv_path = "datasets/simulation_gene_data.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 3.00, 4.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t][["x1", "x2"]].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    

class SimulationDataset_wo_3():
    def __init__(self):
        self.csv_path = "datasets/simulation_gene_data.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00, 4.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t][["x1", "x2"]].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    

class SimulationDataset_wo_4():
    def __init__(self):
        self.csv_path = "datasets/simulation_gene_data.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t][["x1", "x2"]].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    

class EmtDataset_wo_1():
    def __init__(self):
        self.csv_path = "datasets/emt.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 2.00, 3.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t].iloc[:,1:].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
class EmtDataset_wo_2():
    def __init__(self):
        self.csv_path = "datasets/emt.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 3.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t].iloc[:,1:].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
class EmtDataset_wo_3():
    def __init__(self):
        self.csv_path = "datasets/emt.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t].iloc[:,1:].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    

class MouseDataset_wo_1():
    def __init__(self):
        self.csv_path = "datasets/mouse_hematopoiesis.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 2.00,])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t].iloc[:,1:].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
class MouseDataset_wo_2():
    def __init__(self):
        self.csv_path = "datasets/mouse_hematopoiesis.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00,])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t].iloc[:,1:].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (min(batch_size,len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict