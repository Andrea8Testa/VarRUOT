import torch
from geomloss import SamplesLoss

def quantitative_test(
    dataset, 
    results,
    p,
    start_time: float = 0.0,
    end_time: float = 1.0
    ):

    loss = SamplesLoss("sinkhorn", p=p, blur=1e-5, scaling=0.99)
    # loss = SamplesLoss("sinkhorn", p=p)

    results_tensor = results.detach()
    x_pred = results_tensor[..., :-1]
    m_pred = results_tensor[..., -1]
    m_pred_sum = torch.sum(m_pred, dim=-1) / torch.sum(m_pred[0], dim=-1)
    t_pred = torch.linspace(start_time, end_time, x_pred.shape[0])

    all_data = dataset.get_all_particles_batch()
    x_ref = [x for x in all_data.values()]

    m_ref = [data.shape[0] / x_ref[0].shape[0] for data in x_ref]
    
    t_ref = torch.linspace(start_time, end_time, len(x_ref))

    batch_size_max = 2000
    n_tests = 20

    evaluation_pos = []
    evaluation_mass = []
    for idx, t in enumerate(t_ref):
        closest_idx = torch.argmin(torch.abs(t - t_pred))

        ref_weights = (torch.ones(x_ref[idx].shape[0]) / x_ref[idx].shape[0]).to(x_pred.device)
        pred_weights = m_pred[closest_idx] / torch.sum(m_pred[closest_idx])

        n1 = x_pred[closest_idx].shape[0]
        n2 = x_ref[idx].shape[0]
        """slice_size = min(n1, n2, batch_size_max)

        d = 0
        for j in range(n_tests):
            indices_1 = torch.randint(0, n1, (slice_size,))
            indices_2 = torch.randint(0, n2, (slice_size,))
            x1_subset = x_pred[closest_idx][indices_1]
            w1_subset = pred_weights[indices_1]
            x2_subset = x_ref[idx][indices_2]
            w2_subset = ref_weights[indices_2]

            d += loss(w1_subset, x1_subset, w2_subset, x2_subset)
        
        measure_pos = d / n_tests"""
        measure_pos = loss(pred_weights, x_pred[closest_idx], ref_weights, x_ref[idx]) ** (1/ p)
        evaluation_pos.append(measure_pos)

        measure_mass = (m_pred_sum[closest_idx] - m_ref[idx]).pow(2)
        evaluation_mass.append(measure_mass)
    return evaluation_pos, evaluation_mass