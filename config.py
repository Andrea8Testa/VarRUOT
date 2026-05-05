import torch


class Config():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_sample_timepoints_num = 10
    train_PINN_sample_timepoints_num = 10
    prob_loss_kernel_size = 0.35
    prob_sample_size = 128
    pinn_loss_zone_range = 15
    lr = 1e-4
    weight_decay = 0
    lr_decay = 0.97
    gradient_clip = 0.1
    train_epochs = 1000
    save_epochs = 100
    batch_size = 100 # 512
    action_loss_weight = 1
    matching_loss_weight = 16.00
    HJB_loss_weight = 1
    PINN_loss_weight = 0.2
    prob_loss_weight = 24.00
    growth_coeff = 1000.0  # For Veres 200
    sigma = 0.1
    integral_sample_std = 1.5
    ot_use_all_data = False
    ot_sample_size = 2048
    

    valid_batchsize = 100
    valid_timesteps = 20*4
    visualize_timesteps = 200

    dim_reduction_method = "umap"
    # visual

"""class Config():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_sample_timepoints_num = 8
    train_PINN_sample_timepoints_num = 10
    prob_loss_kernel_size = 0.35
    prob_sample_size = 128
    pinn_loss_zone_range = 15
    lr = 1e-4
    weight_decay = 0
    lr_decay = 0.97
    gradient_clip = 0.1
    train_epochs = 1001
    save_epochs = 100
    batch_size = 1024 # 512
    action_loss_weight = 1
    matching_loss_weight = 16.00
    HJB_loss_weight = 1
    PINN_loss_weight = 0.2
    prob_loss_weight = 24.00
    growth_coeff = 50.0  # For Veres 200
    sigma = 0.1
    integral_sample_std = 1.5
    ot_use_all_data = False
    ot_sample_size = 2048
    

    valid_batchsize = 1024
    valid_timesteps = 20*4
    visualize_timesteps = 200

    dim_reduction_method = "umap"
    # visual"""

# class Config():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     train_sample_timepoints_num = 10
#     train_PINN_sample_timepoints_num = 10
#     prob_loss_kernel_size = 0.35
#     prob_sample_size = 128
#     pinn_loss_zone_range = 15
#     lr = 2e-5
#     weight_decay = 0
#     lr_decay = 0.97
#     gradient_clip = 0.1
#     train_epochs = 1001
#     save_epochs = 100
#     batch_size = 512
#     action_loss_weight = 1
#     matching_loss_weight = 16.00
#     HJB_loss_weight = 1
#     PINN_loss_weight = 0.2
#     prob_loss_weight = 24.00
#     growth_coeff = 2.0
#     sigma = 0.1
#     integral_sample_std = 1.5
#     ot_use_all_data = False
#     ot_sample_size = 2048
    

#     valid_batchsize = 512
#     valid_timesteps = 20*4
#     visualize_timesteps = 200

#     dim_reduction_method = "raw"
#     # visual