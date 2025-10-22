import torch 
import torch.optim as optim
from config import Config
from model import MultiplierModel, InteractionModel, LogDistributionModel
from dataset import Gaussian2dDataset, MouseDataset, SimulationDataset, EmtDataset, VeresDataset, MouseHardDataset, Dim50Dataset, Gaussian50Dataset, Gaussian100Dataset
from VarRUOT_WFR import VarRUOT_WFR
from VarRUOT_modified import VarRUOT_modified
import os 

import matplotlib.pyplot as plt

def train(interactionot,  save_path=None):

    os.makedirs("checkpoints", exist_ok=True)
    min_OT_loss = 1e9
    cfg = Config()
    optimizer = optim.AdamW(
        list(interactionot.multiplyer_model.parameters()) , 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay,
    )

    def lr_schedule_lambda(step):
        if step < 3001:

            return cfg.lr_decay ** (step // 10)
        else:

            return 0.00001

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_lambda)


    loss_total = []
    loss_action = []
    loss_matching = []
    loss_HJB = []
    loss_PINN = []
    loss_prob = []

    for epoch in range(cfg.train_epochs):
        optimizer.zero_grad()

        tot_loss, action_loss, matching_loss,ot_loss, HJB_loss, PINN_loss, prob_loss = interactionot.compute_loss()
        tot_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(interactionot.multiplyer_model.parameters()) ,
            max_norm=cfg.gradient_clip 
        )
        optimizer.step()
        scheduler.step()  


        loss_total.append(tot_loss.item())
        loss_action.append(action_loss.item())
        loss_matching.append(matching_loss.item())
        loss_HJB.append(HJB_loss.item())
        loss_PINN.append(PINN_loss.item())
        loss_prob.append(prob_loss.item())

        print(f"Epoch {epoch} | Total Loss: {loss_total[-1]:.4f} | Action Loss: {loss_action[-1]:.4f} | " \
              f"Matching Loss: {loss_matching[-1]:.4f} | HJB Loss: {loss_HJB[-1]:.4f} | " \
              f"PINN Loss: {loss_PINN[-1]:.4f} | Prob Loss: {loss_prob[-1]:.4f} | Min OT Loss: {min_OT_loss:.4f}")
        print("----------------------------------------------------------")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        

        axes[0, 0].plot(range(len(loss_total)), loss_total, marker='o', color='royalblue', alpha=0.7)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)
        

        axes[0, 1].plot(range(len(loss_action)), loss_action, marker='o', color='seagreen', alpha=0.7)
        axes[0, 1].set_title("Action Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True)
        

        axes[1, 0].plot(range(len(loss_matching)), loss_matching, marker='o', color='salmon', alpha=0.7)
        axes[1, 0].set_title("Matching Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True)
        

        axes[1, 1].plot(range(len(loss_HJB)), loss_HJB, marker='o', color='orchid', alpha=0.7)
        axes[1, 1].set_title("HJB Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].grid(True)
        

        axes[2, 0].plot(range(len(loss_PINN)), loss_PINN, marker='o', color='darkorange', alpha=0.7)
        axes[2, 0].set_title("PINN Loss")
        axes[2, 0].set_xlabel("Epoch")
        axes[2, 0].set_ylabel("Loss")
        axes[2, 0].grid(True)
        

        axes[2, 1].plot(range(len(loss_prob)), loss_prob, marker='o', color='teal', alpha=0.7)
        axes[2, 1].set_title("Prob Loss")
        axes[2, 1].set_xlabel("Epoch")
        axes[2, 1].set_ylabel("Loss")
        axes[2, 1].grid(True)

        fig.tight_layout()

        plt.savefig("loss_plot.png")
        plt.close()


        if epoch % cfg.save_epochs == 0:
            torch.save(interactionot.multiplyer_model.state_dict(), f"checkpoints/{save_path}_multiplyer_model_epoch_{epoch}.pt")
        if loss_matching[-1] < min_OT_loss:
            min_OT_loss = loss_matching[-1]
            torch.save(interactionot.multiplyer_model.state_dict(), f"checkpoints/{save_path}_best_multiplyer_model.pt")
    return

if __name__ == "__main__":

    dataset = MouseDataset()
    multipliermodel = MultiplierModel(data_dim=2)
    my_ruot = VarRUOT_WFR(dataset, multiplyer_model=multipliermodel)


    # train(my_ruot, save_path="mouse_easy")
    my_ruot.visualize2dresult()
    my_ruot.visualizedatagrowth()

