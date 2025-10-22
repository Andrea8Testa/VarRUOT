import numpy as np 
import torch 
from config import Config
import pandas as pd 

class Gaussian2dDataset:
    def __init__(self, start_n_samples = 1000, end_n_samples = 2000):
        self.cfg = Config
        self.time_steps = np.array([0.0, 1.0])
        self.max_time = np.max(self.time_steps)
        

        self.data_time0 = 0.05*np.random.randn(start_n_samples, 2)

        self.data_time1 = 0.05*np.random.randn(end_n_samples, 2) + np.array([1.0, 1.0])

    def get_certain_time_data(self, t):
        if t == 0.0:
            return self.data_time0
        elif t == 1.0:
            return self.data_time1
        else:
            raise ValueError(f"Time t={t} not supported in the dataset.")

    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        particles_dict = {}
        batch0 = self.sample_batch(self.get_certain_time_data(0.0), batch_size)
        batch1 = self.sample_batch(self.get_certain_time_data(1.0), batch_size)
        particles_dict[0.0] = batch0.to(self.cfg.device)
        particles_dict[1.0] = batch1.to(self.cfg.device)
        return particles_dict

    def get_all_particles_batch(self):
        particles_dict = {}
        batch0 = torch.tensor(self.get_certain_time_data(0.0), dtype=torch.float32)
        batch1 = torch.tensor(self.get_certain_time_data(1.0), dtype=torch.float32)
        particles_dict[0.0] = batch0.to(self.cfg.device)
        particles_dict[1.0] = batch1.to(self.cfg.device)
        return particles_dict
    
class MouseDataset():
    def __init__(self):
        self.csv_path = "./datasets/mouse_hematopoiesis.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00])
        self.max_time = np.max(self.time_steps)
        
    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t][["x1", "x2"]].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        batch1 = self.sample_batch(self.get_certain_time_data(0.00), batch_size)
        batch2 = self.sample_batch(self.get_certain_time_data(1.00), batch_size)
        batch3 = self.sample_batch(self.get_certain_time_data(2.00), batch_size)

        particles_dict[0.00] = batch1.to(self.cfg.device)
        particles_dict[1.00] = batch2.to(self.cfg.device)
        particles_dict[2.00] = batch3.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        batch1 = torch.tensor(self.get_certain_time_data(0.00), dtype=torch.float)
        batch2 = torch.tensor(self.get_certain_time_data(1.00), dtype=torch.float)
        batch3 = torch.tensor(self.get_certain_time_data(2.00), dtype=torch.float)

        particles_dict[0.00] = batch1.to(self.cfg.device)
        particles_dict[1.00] = batch2.to(self.cfg.device)
        particles_dict[2.00] = batch3.to(self.cfg.device)

        return particles_dict
    
    def sample_validation_particles_batch(self, batch_size):
        validation_batch = torch.tensor(self.get_certain_time_data(2.00), dtype=torch.float ).to(self.cfg.device)
        validation_time = 2.00
        return (validation_time, validation_batch)
    

class SimulationDataset():
    def __init__(self):
        self.csv_path = "datasets/simulation_gene_data.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00, 4.00])
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t][["x1", "x2"]].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randperm(len(x))[:min(batch_size, len(x))]
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            # print(len(this_batch))
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self, ):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict

    
class EmtDataset():
    def __init__(self):
        self.csv_path = "datasets/emt.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00])
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
    

class VeresDataset():
    def __init__(self):
        self.csv_path = "datasets/Veres_alltime.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00])
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
    

class MouseHardDataset():
    def __init__(self):
        self.csv_path = "datasets/Weinreb_alltime_umap.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00,])
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
    


class Dim50Dataset():
    def __init__(self):
        self.csv_path = "datasets/Weinreb_alltime.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, 2.00,])
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
    


class Gaussian20Dataset():
    def __init__(self):
        self.csv_path = "datasets/gaussian_20d.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, ])
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


class Gaussian50Dataset():
    def __init__(self):
        self.csv_path = "datasets/gaussian_50d.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, ])
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


class Gaussian100Dataset():
    def __init__(self):
        self.csv_path = "datasets/gaussian_100d.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0.00, 1.00, ])
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
    

    

    