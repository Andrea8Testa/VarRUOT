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
    

class ZebrafishDataset():
    def __init__(self):
        self.csv_path = "./datasets/Zebrafish.csv"
        self.df = pd.read_csv(self.csv_path)
        self.cfg = Config()
        self.time_steps = np.array([0., 1., 2., 3., 4., 5., 6.])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)
        
    def get_certain_time_data(self, t):
        return self.df[self.df["samples"] == t].iloc[:,1:].values
    
    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)
    
    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict
    
    def get_all_particles_batch(self,):
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.cfg.device)

        return particles_dict


class FFHQDataset():
    def __init__(self, input_data="CHILDREN", target_data="ADULT"):
        latents = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/latents.npy")
        gender = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/gender.npy")
        age = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/age.npy")
        # test_inp_images = np.load("/home/tea1rng/unbalanced_workspace/baselines/LightSB/data/test_images.npy")

        def get_mask(gender, age, category):
            if category == "MAN":
                return (gender == "male").reshape(-1)
            elif category == "WOMAN":
                return (gender == "female").reshape(-1)
            elif category == "ADULT":
                return ((age >= 18) & (age != -1)).reshape(-1)
            elif category == "CHILDREN":
                return ((age < 18) & (age != -1)).reshape(-1)
            else:
                raise ValueError(f"Unknown category {category}")

        x_mask = get_mask(gender, age, input_data)
        y_mask = get_mask(gender, age, target_data)

        self.x_data = torch.tensor(latents[x_mask], dtype=torch.float32)
        self.y_data = torch.tensor(latents[y_mask], dtype=torch.float32)
        self.cfg = Config()
        self.relative_mass = None
        self.time_steps = np.array([0., 1.])
        self.max_time = np.max(self.time_steps)

    def sample_batch(self, x, batch_size):
        indices = torch.randint(0, x.shape[0], (batch_size,))
        return x[indices]

    def sample_particles_batch(self, batch_size):
        data_train = {
            0: self.sample_batch(self.x_data, batch_size).to(self.cfg.device),
            1: self.sample_batch(self.y_data, batch_size).to(self.cfg.device)
        }
        return data_train

    def get_all_particles_batch(self):
        data_train = {
            0: self.x_data.to(self.cfg.device),
            1: self.y_data.to(self.cfg.device)
        }
        return data_train


class UCF101Dataset:
    def __init__(self):
        # self.data_path = "./datasets/ucf101_breaststroke_latents_new.pt"
        self.data_path = "./datasets/ucf101_blowing_latents.pt"
        # self.data = torch.load(self.data_path)  # [N, T, D]
        self.data = torch.load(self.data_path)[9:10, :35]  # [N, T, D]
        # self.data = torch.load(self.data_path)[1:2, :35]  # [N, T, D]
        self.data = self.data[:, ::7]  # [N, T, D]
        print("self.data: ", self.data.shape)
        self.N, self.T, self.D = self.data.shape
        self.cfg = Config()
        self.time_steps = np.arange(self.T, dtype=float)
        self.max_time = np.max(self.time_steps)
        self.relative_mass = None

    def get_certain_time_data(self, t):
        t = int(t)
        return self.data[:, t, :]  # [N, D]

    def sample_batch(self, x, batch_size):
        # indices = torch.randint(0, x.shape[0], (batch_size,))
        # no rep
        # indices = torch.randperm(x.shape[0])[:batch_size]
        return x[:batch_size]

    def sample_particles_batch(self, batch_size):
        particles_dict = {}

        for time in self.time_steps:
            x = self.get_certain_time_data(time)
            batch = self.sample_batch(x, batch_size)
            particles_dict[time] = batch.to(self.cfg.device)

        return particles_dict

    def get_all_particles_batch(self):
        particles_dict = {}

        for time in self.time_steps:
            x = self.get_certain_time_data(time)
            particles_dict[time] = x.to(self.cfg.device)

        return particles_dict