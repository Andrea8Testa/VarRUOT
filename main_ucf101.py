import torch
from simple_parsing import ArgumentParser
from dataset import UCF101Dataset
from model import MultiplierModel
from VarRUOT_WFR import VarRUOT_WFR
from train import train
from quantitative_metrics import quantitative_test
from visualization_utils import visualizeresult
import numpy as np

# Get arguments
parser = ArgumentParser()
parser.add_argument("--name", type=int, default=0)
args = parser.parse_args()
# code = torch.randint(10000, 100000, (1,)).item()
path_name = f"ucf_{args.name}"
path_name_growth = f"ucf_{args.name}_growth"

dataset = UCF101Dataset()
multipliermodel = MultiplierModel(data_dim=4096)
# state_dict = torch.load("checkpoints/ucf_0_best_multiplyer_model.pt")
# multipliermodel.load_state_dict(state_dict)
num_params = sum(p.numel() for p in multipliermodel.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")
my_ruot = VarRUOT_WFR(dataset, multiplyer_model=multipliermodel)
train(my_ruot, save_path=path_name)

x_dict = dataset.get_all_particles_batch()
particles = len(list(x_dict.values()))
x0 = list(x_dict.values())[0][:2000]
n_particles = x0.shape[0]
m0 = torch.ones(n_particles, device=x0.device, dtype=x0.dtype) / n_particles
t_samples = torch.linspace(0, dataset.max_time, 10 * (particles-1),
                           device=x0.device, dtype=x0.dtype)
x_traj, m_traj, _ = my_ruot.integrate_dynamics_plot(x0, m0)

np.save("x_pred.npy", x_traj.cpu().detach().numpy())
np.save("x0.npy", x0.cpu().detach().numpy())
results = torch.cat([x_traj, m_traj.unsqueeze(-1)], dim=-1)
visualizeresult(dataset, results, path_name, "umap", 
                highlight_times=[0.00, 0.25, 0.50, 0.75, 1.00], visualization_batch=750)
my_ruot.visualizedatagrowth(save_path=path_name_growth)
pos_w2, mass = quantitative_test(dataset, results, p=2, end_time=dataset.max_time)
pos_w1, _ = quantitative_test(dataset, results, p=1, end_time=dataset.max_time)

print(results.shape)
metrics = []
for j in range(len(pos_w2)):
    metrics.append({
        "marginal": j,
        "pos_w2": float(pos_w2[j]),
        "pos_w1": float(pos_w1[j]),
        "mass": float(mass[j]),
    })

# Print
for r in metrics:
    print(f"marginal n {r['marginal']}: pos_w2 {r['pos_w2']}, "
        f"pos_w1 {r['pos_w1']}, mass {r['mass']}.")