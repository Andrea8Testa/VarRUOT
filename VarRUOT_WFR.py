from config import Config
import torch 
from torchdiffeq import odeint
from geomloss import SamplesLoss
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
import math
import torchsde
import os
import pandas as pd
import ot



class VarRUOT_WFR():
    def __init__(self, dataset=None, interaction_model=None, multiplyer_model=None, log_prob_model=None):
        self.dataset = dataset
        self.interaction_model = interaction_model
        self.multiplyer_model = multiplyer_model
        self.log_prob_model = log_prob_model
        
        # self.initial_gmm = 
        
        self.cfg = Config()
        self.batch_size = self.cfg.batch_size
        if self.interaction_model != None :
            self.interaction_model.train()
            self.interaction_model.to(self.cfg.device)
        if self.multiplyer_model != None :
            self.multiplyer_model.train() 
            self.multiplyer_model.to(self.cfg.device)
        if self.log_prob_model != None:
            self.log_prob_model.train()
            self.log_prob_model.to(self.cfg.device)
        
        self.matching_loss_calculator = SamplesLoss("sinkhorn", p=2, blur=0.04)

    
    def gaussian_density(self, data, x, sigma, norm_num):
        N, dim = data.shape
        const = 1.0 / ((2 * math.pi * sigma**2) ** (dim / 2))
        diff = x.unsqueeze(1) - data.unsqueeze(0)  
        squared_dist = torch.sum(diff ** 2, dim=-1)   
        exponent = torch.exp(-squared_dist / (2 * sigma**2))
        density_components = const * exponent          
        densities = (1.0 / norm_num) * torch.sum(density_components, dim=1)
        return densities.unsqueeze(1)
    

    def compute_prob_loss(self, sample_range=(-3, 3), sample_num=1000, ):
        sigma = self.cfg.prob_loss_kernel_size
        sampled_data = self.dataset.sample_particles_batch(batch_size = self.cfg.prob_sample_size)
        particles = list(sampled_data.values())
        time_points = list(sampled_data.keys())
        

        n0 = particles[0].shape[0]
        
        prob_loss = 0.0
        num_samples = len(particles)
        
        for i in range(num_samples):

            t_val = torch.tensor(time_points[i], device=self.cfg.device, dtype=torch.float32)

            x_data = particles[i].detach()
            dim = x_data.shape[1]
            

            samples = torch.rand(sample_num, dim, device=self.cfg.device, dtype=torch.float32) * (sample_range[1] - sample_range[0]) + sample_range[0]
            samples.requires_grad_(True)
            

            predicted_log = self.log_prob_model(t_val, samples)  # [sample_num, 1]
            predicted_prob = torch.exp(predicted_log)
            

            true_prob = self.gaussian_density(x_data, samples, sigma, norm_num=n0)  # [sample_num, 1]
            true_log = torch.log(true_prob + 1e-7)  
            

            loss_prob_rand = torch.mean((predicted_prob - true_prob) ** 2)
            
            # print(predicted_log, true_log)
            

            predicted_score = torch.autograd.grad(
                predicted_log.sum(), samples, create_graph=True
            )[0]  # [sample_num, dim]
            # print(predicted_score)
            true_score = torch.autograd.grad(
                true_log.sum(), samples, create_graph=True
            )[0]  # [sample_num, dim]
            # print(true_score)
            loss_grad_rand = torch.sum((predicted_score - true_score) ** 2, dim = 1)
            loss_grad_rand = torch.mean(loss_grad_rand)
            # print(loss_grad_rand)
            

            all_data = x_data.clone().detach().requires_grad_(True)
            
            predicted_log_all = self.log_prob_model(t_val, all_data)  # [N, 1]
            predicted_prob_all = torch.exp(predicted_log_all)
            
            true_prob_all = self.gaussian_density(x_data, all_data, sigma, norm_num=n0)  # [N, 1]
            true_log_all = torch.log(true_prob_all + 1e-7)
            
            loss_prob_data = torch.mean((predicted_prob_all - true_prob_all) ** 2)
            
            predicted_score_all = torch.autograd.grad(
                predicted_log_all.sum(), all_data, create_graph=True
            )[0]  # [N, dim]
            true_score_all = torch.autograd.grad(
                true_log_all.sum(), all_data, create_graph=True
            )[0]  # [N, dim]
            loss_grad_data = torch.sum((predicted_score_all - true_score_all) ** 2, dim = 1)
            loss_grad_data = torch.mean(loss_grad_rand)
            # print(loss_prob_data, loss_grad_data)
            
            total_loss = loss_prob_rand + loss_grad_rand + loss_prob_data + loss_grad_data
            prob_loss += total_loss
            
            # print(f"Time {time_points[i]}: loss_prob_rand = {loss_prob_rand.item():.6f}, loss_grad_rand = {loss_grad_rand.item():.6f}, loss_prob_data = {loss_prob_data.item():.6f}, loss_grad_data = {loss_grad_data.item():.6f}")
        
        prob_loss = prob_loss / num_samples
        return prob_loss



    def compute_vector_field(self, t, x):
        if not x.requires_grad:
            x = x.requires_grad_(True)
    

        output = self.multiplyer_model(t, x)
        

        grad_output = torch.ones_like(output)  
        vector_field = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_output,
            create_graph=True,  
            retain_graph=True,  
            only_inputs=True
        )[0]
        
        return vector_field
    
    def compute_multiplyer_t_derivative_1(self, t, x):

        t = t.clone().detach().requires_grad_(True)
        
        output = self.multiplyer_model(t, x)
        
        grad_list = []

        for i in range(output.shape[0]):

            grad_t = torch.autograd.grad(
                outputs=output[i],
                inputs=t,
                retain_graph=True,
                create_graph=True
            )[0]  
            grad_list.append(grad_t)
        

        t_derivs = torch.stack(grad_list).unsqueeze(1)
        print((t_derivs))
        return t_derivs

    def compute_multiplyer_t_derivative(self, t, x):

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        

        if t.dim() == 0:
            t_tensor = torch.full((x.shape[0], 1), t.item(), device=x.device, dtype=x.dtype, requires_grad=True)
        elif t.dim() == 1:

            t_tensor = t.unsqueeze(1).clone().detach().requires_grad_(True)
        elif t.dim() == 2 and t.size(1) == 1:
            t_tensor = t.clone().detach().requires_grad_(True)
        else:
            raise ValueError("t has invalid shape. Expected shape: [batch_size, 1] or [batch_size].")
        

        output = self.multiplyer_model(t_tensor, x)
        

        t_derivs = torch.autograd.grad(
            outputs=output,
            inputs=t_tensor,
            grad_outputs=torch.ones_like(output),
            create_graph=True
        )[0]
        # print(t_derivs)
        
        return t_derivs
    
    def compute_p_t_derivative(self, t, x):
        batch_size, dim = x.shape
        device = x.device

        x = x.detach() 
        if not t.requires_grad:
            t = t.requires_grad_(True)

        x_detached = x.detach().requires_grad_(False)
        log_rho = self.log_prob_model(t, x_detached)
        rho = torch.exp(log_rho)
        

        dlog_rho_dt = torch.autograd.grad(
            outputs=log_rho,
            inputs=t,
            grad_outputs=torch.ones_like(log_rho),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        

        drho_dt = rho * dlog_rho_dt
        return drho_dt
    

    def compute_continuity_divergence(self, t, x):
        batch_size, dim = x.shape
        device = x.device


        if not x.requires_grad:
            x.requires_grad_()


        log_rho = self.log_prob_model(t, x)          # [batch_size, 1]
        rho = torch.exp(log_rho)                     # [batch_size, 1]
        v   = self.compute_vector_field(t, x)        # [batch_size, dim]
        local_flux = rho * v                         # [batch_size, dim]

        flux = local_flux 


        divergence = 0.0
        for i in range(dim):
            grad_flux_i = torch.autograd.grad(
                outputs=flux[:, i],
                inputs=x,
                grad_outputs=torch.ones_like(flux[:, i]),
                create_graph=True,
                retain_graph=True
            )[0][:, i]  # [batch_size]
            divergence += grad_flux_i


        return divergence.unsqueeze(1)
    

    def compute_continuity_growth_term(self, t, x):
        log_p = self.log_prob_model(t, x)  
        p = torch.exp(log_p)               
        g = self.multiplyer_model(t, x) / self.cfg.growth_coeff 
        if g.ndim > 1:
            g = g.squeeze(-1)             
        p = p.squeeze(-1)
        return p * g
    

    def compute_continuity_noise_term(self, t, m):
        if not m.requires_grad:
            m.requires_grad_(True)
        log_rho = self.log_prob_model(t, m)
        if log_rho.dim() > 1 and log_rho.size(1) == 1:
            log_rho = log_rho.squeeze(1)  
        rho = torch.exp(log_rho)
        grad_log_rho = torch.autograd.grad(
            outputs=log_rho.sum(),
            inputs=m,
            create_graph=True
        )[0]
        A = -0.5 * self.cfg.sigma**2 * rho.unsqueeze(1) * grad_log_rho 
        divergence = 0
        d = m.shape[1]
        for i in range(d):
            grad_A_i = torch.autograd.grad(
                outputs=A[:, i].sum(),
                inputs=m,
                create_graph=True,
                retain_graph=True
            )[0][:, i]  # [batch_size]
            divergence = divergence + grad_A_i
        return divergence.unsqueeze(1)


    def compute_total_continuity(self, t, x):
        residual = (self.compute_p_t_derivative(t, x) +
                    self.compute_continuity_divergence(t, x) -
                    self.compute_continuity_growth_term(t, x) + 
                    self.compute_continuity_noise_term(t, x))
        # print(self.compute_continuity_noise_term(t, x).shape)
        loss = residual ** 2
        loss = torch.sum(loss) / x.shape[0]
        return loss
    

    def compute_pinn_loss(self, sample_num=64, ):
        sample_zone = (-self.cfg.pinn_loss_zone_range, self.cfg.pinn_loss_zone_range)

        t_samples = torch.linspace(
            0, 
            self.dataset.max_time, 
            self.cfg.train_PINN_sample_timepoints_num,
            device=self.cfg.device,
            dtype=torch.float32
        )

        sampled_data = self.dataset.sample_particles_batch(self.batch_size)
        particles = list(sampled_data.values())
        x0 = particles[0].clone().detach()
        dim = x0.shape[-1]
        x_samples = torch.rand(sample_num, dim, device=self.cfg.device, dtype=x0.dtype)* (sample_zone[1] - sample_zone[0]) + sample_zone[0] 
        PINN_loss = 0
        dt = t_samples[1] - t_samples[0]
        for t in t_samples:
            PINN_loss += self.compute_total_continuity(t, x_samples) * dt
        return PINN_loss
    
    

    def compute_interaction_potential(self, x):
        batch_size, embed_dim = x.shape
        diff = x.unsqueeze(1) - x.unsqueeze(0) 
        diff_flat = diff.reshape(-1, embed_dim)
        potentials_flat = self.interaction_model(diff_flat)
        potentials = potentials_flat.view(batch_size, batch_size)
        eye = torch.eye(batch_size, device=x.device).unsqueeze(-1)  
        potentials = potentials * (1 - eye)
        total_potential = torch.triu(potentials, diagonal=1).sum()
        num_pairs = batch_size * (batch_size - 1) / 2
        normalized_potential = total_potential / num_pairs
        return normalized_potential
    
    
    def compute_interaction_force(self, x, m):
        batch_size, embed_dim = x.shape

        diff = x.unsqueeze(1) - x.unsqueeze(0)  # shape: (batch_size, batch_size, embed_dim)
        diff_flat = diff.reshape(-1, embed_dim)
        diff_flat.requires_grad_(True)


        potentials_flat = self.interaction_model(diff_flat)


        grad_potential = torch.autograd.grad(
            outputs=potentials_flat,
            inputs=diff_flat,
            grad_outputs=torch.ones_like(potentials_flat),
            create_graph=True,
        )[0]
        force_flat = -grad_potential


        force_matrix = force_flat.view(batch_size, batch_size, embed_dim)

        eye = torch.eye(batch_size, device=x.device).unsqueeze(-1)
        force_matrix = force_matrix * (1 - eye)


        mass_matrix = m.unsqueeze(0).unsqueeze(-1) 
        weighted_force_matrix = force_matrix * mass_matrix


        net_force_sum = weighted_force_matrix.sum(dim=1)

        normalization = (mass_matrix.expand(batch_size, -1, -1) * (1 - eye)).sum(dim=1)
        normalization = normalization + 1e-8  
        net_force = net_force_sum / normalization

        return net_force
    


    def compute_interaction_force_RBM(self, x, m):
        batch_size, embed_dim = x.shape
        device = x.device


        perm = torch.randperm(batch_size, device=device)
        x_shuffled = x[perm]
        m_shuffled = m[perm]


        if batch_size % 2 == 0:
            num_pairs = batch_size // 2
            num_triples = 0
        else:
            num_pairs = (batch_size - 3) // 2
            num_triples = 1


        net_force = torch.zeros_like(x)


        if num_pairs > 0:

            pairs_x = x_shuffled[:num_pairs * 2].view(num_pairs, 2, embed_dim)
            pairs_m = m_shuffled[:num_pairs * 2].view(num_pairs, 2)
            

            diff_pairs = pairs_x[:, 0] - pairs_x[:, 1]
            diff_pairs.requires_grad_(True)
            

            potentials_pairs = self.interaction_model(diff_pairs)
            grad_potential = torch.autograd.grad(
                outputs=potentials_pairs,
                inputs=diff_pairs,
                grad_outputs=torch.ones_like(potentials_pairs),
                create_graph=True,
            )[0]

            mass_product = (pairs_m[:, 0] * pairs_m[:, 1]).unsqueeze(-1)
            force_pairs = - mass_product * grad_potential
            

            force_assigned = torch.zeros(num_pairs * 2, embed_dim, device=device)
            force_assigned[0::2] = force_pairs      
            force_assigned[1::2] = -force_pairs     
            net_force[perm[:num_pairs * 2]] = force_assigned


        if num_triples > 0:

            triple_x = x_shuffled[num_pairs * 2:].view(3, embed_dim)
            triple_m = m_shuffled[num_pairs * 2:].view(3)
            

            diff_triple = triple_x.unsqueeze(0) - triple_x.unsqueeze(1)

            diff_triple_flat = diff_triple.view(-1, embed_dim)
            diff_triple_flat.requires_grad_(True)
            

            potentials_triple = self.interaction_model(diff_triple_flat)
            grad_potential_triple = torch.autograd.grad(
                outputs=potentials_triple,
                inputs=diff_triple_flat,
                grad_outputs=torch.ones_like(potentials_triple),
                create_graph=True,
            )[0]
            force_triple_flat = -grad_potential_triple  

            force_matrix = force_triple_flat.view(3, 3, embed_dim)
            

            mass_matrix = (triple_m.unsqueeze(0) * triple_m.unsqueeze(1)).unsqueeze(-1)

            force_matrix_weighted = force_matrix * mass_matrix
            

            eye = torch.eye(3, device=device).unsqueeze(-1)  
            force_matrix_weighted = force_matrix_weighted * (1 - eye)

            force_triple = force_matrix_weighted.sum(dim=1) * 2  
            net_force[perm[num_pairs * 2:]] = force_triple

        return net_force
    

    def compute_particle_dynamics(self, x0, m0, t_samples):
        dt = t_samples[-1] - t_samples[0]
        dt = dt / self.cfg.train_sample_timepoints_num

        if m0.ndim == 1:
            m0 = m0.unsqueeze(1)
        

        y0 = torch.cat([x0, m0], dim=1)
        
        class ParticleSDE(torchsde.SDEIto):
            def __init__(self, parent, sigma):

                super().__init__(noise_type="diagonal")
                self.parent = parent
                self.sigma = sigma

                self.x_dim = x0.shape[1]
            
            def f(self, t, y):

                x = y[:, :self.x_dim]
                m = y[:, self.x_dim]  # (n_particles,)
                

                v = self.parent.compute_vector_field(t, x)
                

                g_val = self.parent.multiplyer_model(t, x) / self.parent.cfg.growth_coeff
                if g_val.ndim > 1:
                    g_val = g_val.squeeze(-1)
                dx_dt = v
                dm_dt = g_val * m
                

                return torch.cat([dx_dt, dm_dt.unsqueeze(1)], dim=1)
            
            def g(self, t, y):


                diffusion_x = self.sigma * torch.ones_like(y[:, :self.x_dim])

                diffusion_m = torch.zeros(y.shape[0], 1, device=y.device)

                return torch.cat([diffusion_x, diffusion_m], dim=1)
        

        sde_model = ParticleSDE(self, self.cfg.sigma)
        

        trajectory = torchsde.sdeint(sde_model, y0, t_samples, method='euler', dt=dt)
        

        x_traj = trajectory[..., :x0.shape[1]]
        m_traj = trajectory[..., x0.shape[1]]
        
        return x_traj, m_traj



    def compute_HJB_main_term(self, t, x):
        dphi_dt = self.compute_multiplyer_t_derivative(t, x)
        # dphi_dt = self.compute_multiplyer_t_derivative_1(t, x)
        # print(dphi_dt)
        grad_phi = self.compute_vector_field(t, x)
        grad_sq = 0.5 * torch.sum(grad_phi**2, dim=1, keepdim=True)
        # print(grad_sq.shape)
        output = dphi_dt + grad_sq
        return output
    

    def compute_HJB_growth_term(self, t, x):
        phi = self.multiplyer_model(t, x)
        # print(phi.shape)
        return phi ** 2 / (2 * self.cfg.growth_coeff) 
    

    def compute_HJB_noise_term(self, t, m):
        if not m.requires_grad:
            m.requires_grad_(True)
        phi = self.multiplyer_model(t, m)  
        if phi.dim() > 1 and phi.size(1) == 1:
            phi = phi.squeeze(1) 
        grad_phi = torch.autograd.grad(
            outputs=phi.sum(),  
            inputs=m,
            create_graph=True
        )[0]
        laplacian = 0
        for i in range(m.shape[1]):
            second_deriv = torch.autograd.grad(
                outputs=grad_phi[:, i].sum(),
                inputs=m,
                retain_graph=True
            )[0][:, i]
            laplacian = laplacian + second_deriv
        noise_term = 0.5 * self.cfg.sigma**2 * laplacian
        # print(noise_term)
        return noise_term.unsqueeze(1)

    


    def compute_total_HJB(self, t, x, m):
        batch_size = x.shape[0]
        # print(batch_size)
        output = self.compute_HJB_main_term(t, x) + self.compute_HJB_growth_term(t, x)  + self.compute_HJB_noise_term(t, x)
        # print(output[0])
        # print(self.compute_HJB_noise_term(t, x))
        output = output ** 2 
        # print(m.shape)
        # print(torch.sum(output) / batch_size)
        # output = output.squeeze(1)
        # print(output.shape)
        # print(m.shape)
        # print(self.compute_HJB_main_term(t, x), self.compute_HJB_growth_term(t, x), self.compute_HJB_noise_term(t, x))
        # print(self.compute_HJB_main_term(t, x)[0],self.compute_HJB_growth_term(t, x)[0]  , self.compute_HJB_noise_term(t, x)[0] )
        # print(output**2)
        output = torch.sum(output * m.unsqueeze(1)) / torch.sum(m)
        # output =  torch.sum(output) / batch_size
        # print(output)
        # print(m)
        # print(output)
        return output
    


    def compute_HJB_loss(self, sample_range=(-15, 15), sample_num=64):
        t_samples = torch.linspace(0, self.dataset.max_time, self.cfg.train_PINN_sample_timepoints_num,
                                device=self.cfg.device, dtype=torch.float32)
        
        sampled_data = self.dataset.sample_particles_batch(self.batch_size)
        particles = list(sampled_data.values())
        x0 = particles[0].clone().detach()  
        dim = x0.shape[-1]
        
        x_samples = torch.rand(sample_num, dim, device=self.cfg.device, dtype=x0.dtype)
        x_samples = x_samples * (sample_range[1] - sample_range[0]) + sample_range[0]
        
        dt = t_samples[1] - t_samples[0]
        
        HJB_loss = 0.0
        for t in t_samples:
            HJB_loss += self.compute_total_HJB(t, x_samples) * dt

        return HJB_loss


    

    def compute_matching_loss(self, ):
        use_all_data = self.cfg.ot_use_all_data
        all_data = self.dataset.get_all_particles_batch()
        sampled_data = self.dataset.sample_particles_batch(self.cfg.ot_sample_size)
        sampled_data = list(sampled_data.values())
        sorted_time_points = sorted(all_data.keys())
        x0 = all_data[sorted_time_points[0]].clone().detach()  
        n0 = x0.shape[0]
        m0 = torch.ones(n0, device=x0.device, dtype=x0.dtype)
        m0_total = m0.sum()
        total_loss = 0.0
        total_ot_loss = 0.0
        x_current = x0
        m_current = m0 / m0_total 

        for i in range(1, len(sorted_time_points)):
            t_start = sorted_time_points[i - 1]
            t_end = sorted_time_points[i]
            t_samples = torch.linspace(t_start, t_end,
                                    self.cfg.train_sample_timepoints_num,
                                    device=x0.device,
                                    dtype=x0.dtype)
            x_seg_traj, m_seg_traj = self.compute_particle_dynamics(x_current, m_current, t_samples) 
            
            x_sim = x_seg_traj[-1] 
            m_sim = m_seg_traj[-1]  

            if use_all_data:
                x_true = all_data[t_end].clone().detach()
                n_data = x_true.shape[0]

                # x_sample = all_data[t_end].clone().detach()
                weights_true = torch.ones(x_true.shape[0], device=x_true.device, dtype=x_true.dtype) 
                weights_true /= torch.sum(weights_true)
                weights_sim = m_sim / torch.sum(m_sim)
                ot_loss = self.matching_loss_calculator(weights_true, x_true, weights_sim, x_sim)

                sim_mass_ratio = m_sim.sum() 
                true_mass_ratio = n_data / n0
                # print(true_mass_ratio, sim_mass_ratio)
                mass_ratio_loss = (sim_mass_ratio - true_mass_ratio) ** 2

            else :
                x_true = all_data[t_end].clone().detach() 
                x_sample = sampled_data[i].clone().detach()
                n_data = x_true.shape[0]

                # x_sample = all_data[t_end].clone().detach()
                weights_true = torch.ones(x_sample.shape[0], device=x_sample.device, dtype=x_sample.dtype) 
                weights_true /= torch.sum(weights_true)
                weights_sim = m_sim / torch.sum(m_sim)
                ot_loss = self.matching_loss_calculator(weights_true, x_sample, weights_sim, x_sim)

                sim_mass_ratio = m_sim.sum()  
                true_mass_ratio = n_data / n0
                # print(true_mass_ratio, sim_mass_ratio)
                mass_ratio_loss = (sim_mass_ratio - true_mass_ratio) ** 2

            # print(ot_loss, mass_ratio_loss)


            total_loss += ot_loss + mass_ratio_loss
            total_ot_loss += ot_loss.item()
            x_current = x_sim
            m_current = m_sim 
        return total_loss, total_ot_loss



    def compute_action_loss(self):

        sampled_data = self.dataset.sample_particles_batch(self.batch_size)
        particles = list(sampled_data.values())
        x0 = particles[0].clone().detach()  
        n_particles = x0.shape[0]

        m0 = torch.ones(n_particles, device=x0.device, dtype=x0.dtype) / n_particles


        t_samples = torch.linspace(
            0,
            self.dataset.max_time,
            self.cfg.train_sample_timepoints_num * (len(particles)-1),
            device=self.cfg.device,
            dtype=x0.dtype
        )
        x_traj, m_traj = self.compute_particle_dynamics(x0, m0, t_samples)
        dt = t_samples[1] - t_samples[0]

        action_loss = 0.0
        HJB_loss = 0.0

        for i in range(len(t_samples)):
            this_t = t_samples[i]
            this_x = x_traj[i]
            this_m = m_traj[i]
            this_vel = self.compute_vector_field(this_t, this_x)
            loss_vel = 0.5 * torch.sum((this_vel ** 2).sum(dim=1) * this_m) 
            this_g = self.multiplyer_model(this_t, this_x) / self.cfg.growth_coeff 
            if this_g.ndim > 1:
                this_g = this_g.squeeze(-1)
            loss_g = self.cfg.growth_coeff * 0.5 * torch.sum((this_g ** 2) * this_m)

            action_loss += (loss_vel + loss_g) * dt
            HJB_loss += self.compute_total_HJB(this_t, this_x, this_m) * dt
            # print(HJB_loss)

        return action_loss, HJB_loss
    



    def compute_loss(self, ):
        action_loss_weight = self.cfg.action_loss_weight
        matching_loss_weight = self.cfg.matching_loss_weight
        HJB_loss_weight = self.cfg.HJB_loss_weight
        PINN_loss_weight = self.cfg.PINN_loss_weight
        prob_loss_weight = self.cfg.prob_loss_weight


        action_loss = torch.tensor(0)
        matching_loss = torch.tensor(0)
        HJB_loss = torch.tensor(0)
        PINN_loss = torch.tensor(0)
        prob_loss = torch.tensor(0)
        
        action_loss, HJB_loss = self.compute_action_loss()
        matching_loss, ot_loss = self.compute_matching_loss() 
        # HJB_loss = self.compute_HJB_loss()
        # PINN_loss = self.compute_pinn_loss()
        # prob_loss = self.compute_prob_loss()
        # print(initial_loss)

        # print(f"action loss : {action_loss}")
        # print(f"particle matching loss : {matching_loss}")
        # print(f"HJB loss : {HJB_loss}")
        # print(f"PINN loss : {PINN_loss}")
        # print(f"prob net matching loss : {prob_loss}")

        tot_loss = action_loss_weight * action_loss + matching_loss_weight * matching_loss + HJB_loss_weight * HJB_loss + PINN_loss_weight * PINN_loss + prob_loss_weight * prob_loss
        return tot_loss, action_loss, matching_loss, ot_loss, HJB_loss, PINN_loss, prob_loss
    

    def integrate_dynamics_plot(self, x0, m0):

        def dynamics(t, state):
            x, m = state  # x shape: [batch, orig_dim] ; m shape: [batch]

            dx_dt = self.compute_vector_field(t, x)

            g_val = self.multiplyer_model(t, x) / self.cfg.growth_coeff
            if g_val.ndim > 1:
                g_val = g_val.squeeze(-1)
            dm_dt = g_val * m
            return dx_dt, dm_dt

        t_samples = torch.linspace(
            0, self.dataset.max_time, self.cfg.visualize_timesteps,
            device=self.cfg.device, dtype=x0.dtype
        )
        
        traj_x = [x0]  
        traj_m = [m0] 
        state = (x0, m0)  
        

        for i in range(len(t_samples) - 1):
            t = t_samples[i]
            h = t_samples[i+1] - t_samples[i]  
            
            x_curr, m_curr = state
            x_curr = x_curr.requires_grad_(True)
            m_curr = m_curr.requires_grad_(True)
            

            dx_dt, dm_dt = dynamics(t, (x_curr, m_curr))

            noise = torch.randn_like(x_curr) * torch.sqrt(h)
            next_x = x_curr + h * dx_dt + self.cfg.sigma * noise

            next_m = m_curr + h * dm_dt
            

            state = (next_x.detach(), next_m.detach())
            traj_x.append(state[0])
            traj_m.append(state[1])
        
        traj_x = torch.stack(traj_x, dim=0)  # shape: [timesteps, batch, orig_dim]
        traj_m = torch.stack(traj_m, dim=0)  # shape: [timesteps, batch]

        return traj_x, traj_m, t_samples


    def visualize2dresult(self, save_path=None, highlight_times=None):

        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from sklearn.decomposition import PCA


        sampled_data = self.dataset.sample_particles_batch(self.cfg.valid_batchsize)
        print(sampled_data[0].shape[0])
        particles = list(sampled_data.values())
        x0 = particles[0].clone().detach()
        

        m0 = torch.ones(x0.shape[0], device=x0.device, dtype=x0.dtype)
        m0 = m0 / m0.sum()  
        
        traj_x, traj_m, t_samples = self.integrate_dynamics_plot(x0, m0)

        x_traj_np = traj_x.cpu().detach().numpy()
        m_traj_np = traj_m.cpu().detach().numpy() * sampled_data[0].shape[0]

        all_data = self.dataset.get_all_particles_batch()
        all_particles = list(all_data.values())
        
        num_timesteps, batch_size, orig_dim = x_traj_np.shape


        dim_method = getattr(self.cfg, 'dim_reduction_method', 'pca')  # 'pca', 'umap', or 'both'
        
        if orig_dim > 2:
            x_traj_flat = x_traj_np.reshape(-1, orig_dim)

            sample_list = []
            for key in all_data:
                sample_list.append(all_data[key])
            sample_all = torch.cat(sample_list, dim=0)
            sample_np = sample_all.cpu().detach().numpy()
            
            if dim_method == 'umap':
                try:
                    import umap
                except ImportError:
                    raise ImportError("please install umap-learn to use UMAP for dimensionality reduction")
                reducer = umap.UMAP(n_components=2)
                reducer.fit(sample_np)
                x_traj_flat_2d = reducer.transform(x_traj_flat)
                x_traj_2d = x_traj_flat_2d.reshape(num_timesteps, batch_size, 2)
                reducer_obj = reducer
            elif dim_method == 'both':
                try:
                    import umap
                except ImportError:
                    raise ImportError("please install umap-learn to use UMAP for dimensionality reduction")
                pca_reducer = PCA(n_components=2)
                umap_reducer = umap.UMAP(n_components=2)
                pca_reducer.fit(sample_np)
                umap_reducer.fit(sample_np)
                x_traj_flat_2d_pca = pca_reducer.transform(x_traj_flat)
                x_traj_flat_2d_umap = umap_reducer.transform(x_traj_flat)
                x_traj_2d_pca = x_traj_flat_2d_pca.reshape(num_timesteps, batch_size, 2)
                x_traj_2d_umap = x_traj_flat_2d_umap.reshape(num_timesteps, batch_size, 2)
                reducer_pca = pca_reducer
                reducer_umap = umap_reducer
            else:

                pca_reducer = PCA(n_components=2)
                pca_reducer.fit(sample_np)
                x_traj_flat_2d = pca_reducer.transform(x_traj_flat)
                x_traj_2d = x_traj_flat_2d.reshape(num_timesteps, batch_size, 2)
                reducer_obj = pca_reducer
        else:
            x_traj_2d = x_traj_np  


        global_mass_min = m_traj_np.min()
        global_mass_max = m_traj_np.max()
        global_norm = plt.Normalize(global_mass_min, global_mass_max)


        if orig_dim > 2 and dim_method == 'both':

            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            ax_list = axs
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax_list = [ax]


        if orig_dim > 2:
            if dim_method == 'umap':
                traj_2d_dict = {'umap': x_traj_2d}
            elif dim_method == 'both':
                traj_2d_dict = {'pca': x_traj_2d_pca, 'umap': x_traj_2d_umap}
            else:
                traj_2d_dict = {'pca': x_traj_2d}
        else:
            traj_2d_dict = {'': x_traj_2d}


        for idx, (key_method, x_traj_2d_curr) in enumerate(traj_2d_dict.items()):
            current_ax = ax_list[idx]

            if batch_size >= 20:
                subsample_indices = np.arange(0, batch_size, 20)
            else:
                subsample_indices = np.arange(0, batch_size)

            for i in subsample_indices:

                points = x_traj_2d_curr[:, i, :].reshape(-1, 1, 2)

                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                masses = (m_traj_np[:-1, i] + m_traj_np[1:, i]) / 2.0


                lc = LineCollection(segments, cmap='viridis', norm=global_norm, alpha=0.8)
                lc.set_array(masses)
                lc.set_linewidth(2)
                current_ax.add_collection(lc)


            for particle in all_particles:
                particle_np = particle.cpu().detach().numpy()  
                if particle_np.shape[-1] > 2:

                    if dim_method == 'umap':
                        particle_2d = reducer_obj.transform(particle_np)
                    elif dim_method == 'both':
                        if key_method == 'pca':
                            particle_2d = reducer_pca.transform(particle_np)
                        else:
                            particle_2d = reducer_umap.transform(particle_np)
                    else:
                        particle_2d = reducer_obj.transform(particle_np)
                else:
                    particle_2d = particle_np
                current_ax.scatter(particle_2d[:, 0], particle_2d[:, 1],
                            color="grey", alpha=0.4, s=30, marker="+")


            if highlight_times is not None and len(highlight_times) > 0:

                t_samples_np = t_samples.cpu().detach().numpy()

                cmap_time = plt.cm.get_cmap('tab10')
                colors = [cmap_time(i) for i in range(len(highlight_times))]
                for idx_h, t_high in enumerate(highlight_times):

                    closest_idx = np.argmin(np.abs(t_samples_np - t_high))
                    current_ax.scatter(
                        x_traj_2d_curr[closest_idx, :, 0],
                        x_traj_2d_curr[closest_idx, :, 1],
                        color=colors[idx_h],
                        s=30,
                        marker='o',
                        label=f"t = {highlight_times[idx_h]:.2f}",
                        alpha=0.6
                    )


            if dim_method == 'pca':
                current_ax.set_xlabel("PC1")
                current_ax.set_ylabel("PC2")
            elif dim_method == 'umap':
                current_ax.set_xlabel("UMAP1")
                current_ax.set_ylabel("UMAP2")

            current_ax.legend(fontsize=18)

            current_ax.set_xticks([])
            current_ax.set_yticks([])


            current_ax.set_xlabel("")
            current_ax.set_ylabel("")



            cbar = fig.colorbar(lc, ax=current_ax)

        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            os.makedirs("figs", exist_ok=True)
            path = os.path.join("figs", save_path + ".jpg")
            plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)


    def visualize_vector_field(self, save_path=None, X_umap=None, 
                                  neighbors=30, n_jobs=16, density=3, figsize=(8, 6), 
                                  title='', palette='tab10'):

            import anndata
            import scvelo as scv
            import scanpy as sc

            t_list = self.dataset.time_steps
            data_list = []
            gradients_list = []
            times_list = []
            for t in t_list:
                pts = self.dataset.get_certain_time_data(t)  
                data_list.append(pts)
                times_list.append(np.full((pts.shape[0],), t))

                data_tensor = torch.tensor(pts, device=self.cfg.device, dtype=torch.float32, requires_grad=True)
                dx_dt_tensor = self.compute_vector_field(t, data_tensor)
                gradients_list.append(dx_dt_tensor.detach().cpu().numpy())

            all_data = np.concatenate(data_list, axis=0)
            gradients = np.concatenate(gradients_list, axis=0)
            times_all = np.concatenate(times_list, axis=0)
            

            orig_dim = all_data.shape[1]
            if X_umap is None:
                if orig_dim > 2:
                    dim_method = getattr(self.cfg, 'dim_reduction_method', 'pca')
                    if dim_method == 'umap':
                        try:
                            import umap
                        except ImportError:
                            raise ImportError("please install umap-learn to use UMAP for dimensionality reduction")
                        reducer = umap.UMAP(n_components=2, random_state=42)
                        X_umap = reducer.fit_transform(all_data)
                    else:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        X_umap = pca.fit_transform(all_data)
                else:
                    X_umap = all_data
            

            adata = anndata.AnnData(X=all_data)

            adata.layers['Ms'] = all_data.copy()

            adata.layers['velocity'] = gradients.copy()
            print("velocity shape:", adata.layers['velocity'].shape)

            adata.obsm['X_umap'] = X_umap.copy()
            print("UMAP embedding shape:", adata.obsm['X_umap'].shape)

            adata.obs['time'] = times_all
            adata.obs['time_categorical'] = pd.Categorical(adata.obs['time'])
            

            sc.pp.neighbors(adata, n_neighbors=neighbors, use_rep='X')

            scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=n_jobs)

            scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')
            

            scv.settings.set_figure_params('scvelo')
            scv.pl.velocity_embedding_stream(
                adata,
                basis='umap',
                color='time_categorical',
                figsize=figsize,
                density=density,
                title=title,
                legend_loc='right',
                palette=palette,
                save=   save_path ,
                show=False,
            )

    
    def visualize_raw_data(self, save_path=None):

        import os
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA


        t_samples = self.dataset.time_steps


        data_list = []
        for t in t_samples:
            data_np_array = self.dataset.get_certain_time_data(t)
            particle_data = data_np_array
            data_list.append(torch.tensor(particle_data))
        
        

        traj_x = torch.cat(data_list, dim=0)
        nn, orig_dim = traj_x.shape


        traj_x_np = traj_x.cpu().numpy()
        if orig_dim > 2:

            x_traj_flat = traj_x_np.reshape(-1, orig_dim)
            pca = PCA(n_components=2)
            pca.fit(x_traj_flat)
            data_2d = [] 
            for item in data_list:
                item_np = item.cpu().numpy()
                item_2d = pca.transform(item_np.reshape(-1, orig_dim))
                data_2d.append(item_2d)
        else:
            data_2d = data_list


        fig, ax = plt.subplots(figsize=(8, 6))
        

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        

        for i, item in enumerate(data_2d):
            ax.scatter(
                item[:, 0],
                item[:, 1],
                color=colors[i],
                s=30,
                label=f"t = {t_samples[i].item():.2f}",
                alpha=0.6
            )


        if orig_dim > 2:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        else:
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")


        ax.legend()
        plt.tight_layout()


        if save_path is None:
            plt.show()
        else:
            os.makedirs("figs", exist_ok=True)
            path = os.path.join("figs", save_path + ".jpg")
            plt.savefig(path, dpi=300)

    def visualize2dgrowth(self, grid_range=(-1.5, 1.5), num_points=50, t=0):

        x_vals = np.linspace(grid_range[0], grid_range[1], num_points)
        y_vals = np.linspace(grid_range[0], grid_range[1], num_points)
        XX, YY = np.meshgrid(x_vals, y_vals)
        grid_points_2d = np.stack([XX.ravel(), YY.ravel()], axis=1)

        sampled_data = self.dataset.sample_particles_batch(self.cfg.valid_batchsize)
        sample_x = list(sampled_data.values())[0]  # shape: [batch_size, orig_dim]
        orig_dim = sample_x.shape[-1]

        if orig_dim > 2:

            all_data = self.dataset.get_all_particles_batch()
            sample_list = []
            for key in all_data:
                sample_list.append(all_data[key])
            sample_all = torch.cat(sample_list, dim=0)
            sample_np = sample_all.cpu().detach().numpy()
            pca = PCA(n_components=2)
            pca.fit(sample_np)
            high_dim_grid = pca.inverse_transform(grid_points_2d)
        else:
            pca = None
            high_dim_grid = grid_points_2d

        grid_tensor = torch.tensor(high_dim_grid, dtype=torch.float32, device=self.cfg.device)
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=grid_tensor.dtype, device=self.cfg.device)
            growth = self.multiplyer_model(t_tensor, grid_tensor) / self.cfg.growth_coeff
        growth = growth.view(XX.shape).cpu().numpy()

        plt.figure(figsize=(6, 5))
        cp = plt.contourf(XX, YY, growth, levels=50, cmap='jet')
        plt.contour(XX, YY, growth, levels=15, colors='k', linewidths=0.5)
        if orig_dim > 2:
            plt.title("Growth (PCA Projection)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
        else:
            plt.title("Growth")
            plt.xlabel("x")
            plt.ylabel("y")
        plt.colorbar(cp, label="growth")
        plt.tight_layout()
        plt.show()

    def visualizedatagrowth(self, save_path=None):

        import os
        import torch
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA


        all_data = self.dataset.get_all_particles_batch()
        all_time = list(all_data.keys())
        particle_list = []
        growth_list = []


        for t in all_time:
            particles = all_data[t]  
            t_tensor = torch.tensor(t, dtype=particles.dtype, device=particles.device)
            this_growth = self.multiplyer_model(t_tensor, particles) / self.cfg.growth_coeff  
            particle_list.append(particles)
            growth_list.append(this_growth)


        ps = torch.cat(particle_list, dim=0).detach().cpu().numpy()  
        gs = torch.cat(growth_list, dim=0).detach().cpu().numpy()     

        orig_dim = ps.shape[-1]

        dim_method = getattr(self.cfg, 'dim_reduction_method', 'pca')


        if orig_dim > 2:
            if dim_method == 'umap':
                try:
                    import umap
                except ImportError:
                    raise ImportError("please install umap-learn to use UMAP for dimensionality reduction")
                reducer = umap.UMAP(n_components=2)
                ps_2d = reducer.fit_transform(ps)
                xlabel, ylabel = "UMAP-1", "UMAP-2"
            elif dim_method == 'both':
                try:
                    import umap
                except ImportError:
                    raise ImportError("please install umap-learn to use UMAP for dimensionality reduction")

                pca = PCA(n_components=2)
                ps_2d_pca = pca.fit_transform(ps)

                umap_reducer = umap.UMAP(n_components=2)
                ps_2d_umap = umap_reducer.fit_transform(ps)
                

                fig, axs = plt.subplots(1, 2, figsize=(16, 6))

                sc1 = axs[0].scatter(ps_2d_pca[:, 0], ps_2d_pca[:, 1], c=gs, cmap='viridis', s=30, alpha=0.3)
                axs[0].set_xlabel("PC1")
                axs[0].set_ylabel("PC2")
                axs[0].set_title("Data Growth (PCA Projection)")
                fig.colorbar(sc1, ax=axs[0])

                sc2 = axs[1].scatter(ps_2d_umap[:, 0], ps_2d_umap[:, 1], c=gs, cmap='viridis', s=30, alpha=0.3)
                axs[1].set_xlabel("UMAP-1")
                axs[1].set_ylabel("UMAP-2")
                axs[1].set_title("Data Growth (UMAP Projection)")
                fig.colorbar(sc2, ax=axs[1])
                plt.tight_layout()
                
                if save_path is None:
                    plt.show()
                else:
                    os.makedirs("figs", exist_ok=True)
                    path = os.path.join("figs", save_path + ".jpg")
                    plt.savefig(path, dpi=300)
                return  
            else:
                pca = PCA(n_components=2)
                ps_2d = pca.fit_transform(ps)
                xlabel, ylabel = "PC1", "PC2"
        else:
            ps_2d = ps
            xlabel, ylabel = "x", "y"


        xs = ps_2d[:, 0]
        ys = ps_2d[:, 1]

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(xs, ys, c=gs, cmap='viridis', s=30, alpha=0.3)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])


        plt.xlabel("")
        plt.ylabel("")
        plt.colorbar(sc)
        plt.tight_layout()


        if save_path is None:
            plt.show()
        else:
            os.makedirs("figs", exist_ok=True)
            path = os.path.join("figs", save_path + ".jpg")
            plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)


    def visualizedatagrowth_by_time(self, save_path=None, arrow_sample_ratio=0.1):
       
        all_data = self.dataset.get_all_particles_batch()
        all_time = list(all_data.keys())
        
        for t in all_time:

            particles = all_data[t]

            t_tensor = torch.tensor(t, dtype=particles.dtype, device=particles.device)
            

            this_growth = self.multiplyer_model(t_tensor, particles) / self.cfg.growth_coeff
            

            raw_vector_field = self.compute_vector_field(t_tensor, particles)
            

            norms = torch.norm(raw_vector_field, dim=1, keepdim=True)
            eps = 1e-5  
            max_norm = norms.max()
            

            vector_directions = raw_vector_field / (norms + eps)

            arrow_constant = 0.15

            arrow_lengths = arrow_constant * (norms / (max_norm + eps))

            vector_field = vector_directions * arrow_lengths
            

            ps = particles.detach().cpu().numpy()       
            growth_values = this_growth.detach().cpu().numpy().flatten()  
            vs = vector_field.detach().cpu().numpy()     
            
            orig_dim = ps.shape[-1]
            if orig_dim > 2:

                pca = PCA(n_components=2)
                ps_2d = pca.fit_transform(ps)

                vs_2d = np.dot(vs, pca.components_.T)
                xlabel, ylabel = "PC1", "PC2"
            else:
                ps_2d = ps
                vs_2d = vs
                xlabel, ylabel = "x", "y"
            
            xs = ps_2d[:, 0]
            ys = ps_2d[:, 1]
            
            plt.figure(figsize=(8, 6))

            sc = plt.scatter(xs, ys, c=growth_values, cmap='viridis', s=30, alpha=0.3)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks([])
            plt.yticks([])


            plt.xlabel("")
            plt.ylabel("")
            plt.colorbar(sc)
            

            total_points = ps_2d.shape[0]
            sample_size = int(total_points * arrow_sample_ratio)
            if sample_size > 0:
                indices = np.random.choice(total_points, size=sample_size, replace=False)
                x_sample = xs[indices]
                y_sample = ys[indices]
                dx_sample = vs_2d[indices, 0]
                dy_sample = vs_2d[indices, 1]
                plt.quiver(x_sample, y_sample, dx_sample, dy_sample, 
                        angles='xy', scale_units='xy', scale=1,
                        color='black', alpha=0.4)
            
            plt.tight_layout()
            if save_path is None:
                plt.show()
            else:
                os.makedirs("figs", exist_ok=True)
                path = os.path.join("figs", f"{save_path}_time_{t}.jpg")
                plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()



    

    
    def visualize_gaussian_density(self, sample_range=(-3, 3), num_points=100, sigma=0.2):
        sampled_data = self.dataset.get_all_particles_batch()
        particles = list(sampled_data.values())
        time_points = list(sampled_data.keys())
        

        t0 = None
        t1 = None
        for t in time_points:
            if np.isclose(t, 0.0, atol=1e-3):
                t0 = t
            if np.isclose(t, 1.0, atol=1e-3):
                t1 = t
        if t0 is None or t1 is None:
            print("No t=0 or t=1 found in the data.")
            t0 = time_points[0]
            t1 = time_points[1] if len(time_points) > 1 else time_points[0]
        
        data_t0 = sampled_data[t0].cpu().detach()
        data_t1 = sampled_data[t1].cpu().detach()
        

        x_vals = np.linspace(sample_range[0], sample_range[1], num_points)
        y_vals = np.linspace(sample_range[0], sample_range[1], num_points)
        XX, YY = np.meshgrid(x_vals, y_vals)
        grid_points = np.stack([XX.ravel(), YY.ravel()], axis=1)
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=data_t0.device)
        

        dens_t0 = self.gaussian_density(data_t0, grid_tensor, sigma)
        dens_t1 = self.gaussian_density(data_t1, grid_tensor, sigma)
        
        dens_t0 = dens_t0.view(XX.shape).cpu().numpy()
        dens_t1 = dens_t1.view(XX.shape).cpu().numpy()
        

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cp0 = axes[0].contourf(XX, YY, dens_t0, levels=50, cmap='jet')
        axes[0].contour(XX, YY, dens_t0, levels=15, colors='k', linewidths=0.5)
        axes[0].set_title(f'Gaussian Density at t = {t0}')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        fig.colorbar(cp0, ax=axes[0], label='Density')
        
        cp1 = axes[1].contourf(XX, YY, dens_t1, levels=50, cmap='jet')
        axes[1].contour(XX, YY, dens_t1, levels=15, colors='k', linewidths=0.5)
        axes[1].set_title(f'Gaussian Density at t = {t1}')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        fig.colorbar(cp1, ax=axes[1], label='Density')
        
        plt.tight_layout()
        plt.show()

    def save_simulation_results(self, save_path, run_times, time_points):

        import os
        import numpy as np
        import torch


        output_dir = "output_data"
        os.makedirs(output_dir, exist_ok=True)


        dt = 0.001
        max_time = self.dataset.max_time

        num_steps = int(max_time / dt) + 1


        sim_times = torch.arange(0, max_time + dt / 2, dt, device=self.cfg.device, dtype=torch.float32)

        for run in range(run_times):

            sampled_data = self.dataset.sample_particles_batch(self.cfg.valid_batchsize)
            particles = list(sampled_data.values())
            x0 = particles[0].clone().detach()  
            batch_size, dim = x0.shape


            m0 = torch.ones(batch_size, device=x0.device, dtype=x0.dtype) / batch_size


            def dynamics(t, state):
                x, m = state  # x: [batch_size, dim] ; m: [batch_size]
                dx_dt = self.compute_vector_field(t, x)
                g_val = self.multiplyer_model(t, x) / self.cfg.growth_coeff
                if g_val.ndim > 1:
                    g_val = g_val.squeeze(-1)
                dm_dt = g_val * m
                return dx_dt, dm_dt


            traj_x = [x0]
            traj_m = [m0]
            state = (x0, m0)

            for i in range(num_steps - 1):
                current_t = sim_times[i]

                x_curr, m_curr = state

                x_curr = x_curr.requires_grad_(True)
                m_curr = m_curr.requires_grad_(True)

                dx_dt, dm_dt = dynamics(current_t, (x_curr, m_curr))

                noise = torch.randn_like(x_curr) * torch.sqrt(torch.tensor(dt, dtype=x_curr.dtype, device=x_curr.device))
                next_x = x_curr + dt * dx_dt + self.cfg.sigma * noise
                next_m = m_curr + dt * dm_dt

                state = (next_x.detach(), next_m.detach())
                traj_x.append(state[0])
                traj_m.append(state[1])


            traj_x_tensor = torch.stack(traj_x, dim=0)  # [num_steps, batch_size, dim]
            traj_m_tensor = torch.stack(traj_m, dim=0)      # [num_steps, batch_size]

            traj_m_tensor = traj_m_tensor.unsqueeze(-1)


            saved_indices = []
            for tp in time_points:
                idx = int(round(tp / dt))
                if idx < 0 or idx >= num_steps:
                    raise ValueError(f"time point {tp} is out of simulation range [0, {max_time}]")
                saved_indices.append(idx)

            saved_indices = sorted(set(saved_indices))

            saved_positions = traj_x_tensor[saved_indices, :, :].cpu().detach().numpy()  # [len(time_points), batch_size, dim]
            saved_weights = traj_m_tensor[saved_indices, :, :].cpu().detach().numpy()     # [len(time_points), batch_size, 1]

            pos_filename = os.path.join(output_dir, f"{save_path}_position_time_{run}.npy")
            weight_filename = os.path.join(output_dir, f"{save_path}_weight_time_{run}.npy")
            np.save(pos_filename, saved_positions)
            np.save(weight_filename, saved_weights)

            print(f"Run {run}: Saved position data to {pos_filename} and weight data to {weight_filename}")

    