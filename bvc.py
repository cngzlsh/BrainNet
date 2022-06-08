import torch
import torch.nn as nn
import torch.distributions as dist
from utils import plot_bvc_firing_field

torch.manual_seed(1234)

class BVC:
    '''
    A boundary vector cell with preferred distance and angle
    '''
    def __init__(self, r, theta, sigma_rad, sigma_ang, scaling_factor=1) -> None:
        self.r = torch.Tensor([r])
        self.theta = torch.Tensor([theta])
        self.sigma_rad = sigma_rad # can be a lambda function
        self.sigma_ang = sigma_ang # can be a lambda function
        self.scaling_factor = scaling_factor

    def obtain_firing_rate(self, d, phi):
        '''
        Computes the firing of BVC given a current distance and angle to a boundary.
        Firing rate is proportional to product of two gaussians centerred at the preferred distance and angle
        Vectorised: d and phi can be arrays or matricesd
        '''
        if isinstance(self.sigma_rad, torch.Tensor):
            sigma_rad = self.sigma_rad
        else:
            sigma_rad = torch.Tensor([self.sigma_rad(d)])
        
        if isinstance(self.sigma_ang, torch.Tensor):
            sigma_ang = self.sigma_ang
        else:
            sigma_ang = torch.Tensor([self.sigma_rad(phi)])

        unscaled_firing_rate = torch.exp(-(self.r - d) ** 2 / (2 * sigma_rad ** 2))/ \
                torch.sqrt(2 * torch.pi * sigma_rad ** 2) * \
                    torch.exp(-(self.theta - phi) ** 2 / (2 * sigma_ang ** 2))/ \
                        torch.sqrt(2 * torch.pi * sigma_ang ** 2)
        
        return unscaled_firing_rate * self.scaling_factor
    

class BVCNetwork:
    '''
    A network of boundary vector cells
    Output is the thresholded firing sum, each at the same allocentric reference frame, scaled by a coefficient A
    '''
    def __init__(self, BVCs: list[BVC], coeff, threshold, non_linearity = nn.ReLU()) -> None:
        self.n_cells = len(BVCs)
        self.BVCs = BVCs
        self.coeff = coeff
        self.threshold = threshold
        self.non_linearity = non_linearity
    
    def obtain_firing_rate(self, d, phi):
        firing_rates = torch.stack([bvc.obtain_firing_rate(d=d, phi=phi) for bvc in self.BVCs], dim=0)
        thresholded_rate = torch.sum(firing_rates, dim=0) - self.threshold
        return self.coeff * self.non_linearity(thresholded_rate)


if __name__ == '__main__':
    n_cells = 8             # number of BVCs to simulate
    n_data_points = 10000   # number of data points to generate

    # BVC preferred distances ~ Uniform(0, 10)
    preferred_distances = dist.uniform.Uniform(low=-0, high=10).sample(torch.Size([n_cells]))
    # BVC preferred angles ~ Uniform(-pi, pi)
    preferred_orientations = dist.uniform.Uniform(low=-torch.pi, high=torch.pi).sample(torch.Size([n_cells]))
    sigma_rads = torch.ones(n_cells)
    sigma_angs = torch.ones(n_cells)

    # initialise BVCS and network
    BVCs = [BVC(r=preferred_distances[i],theta=preferred_orientations[i], sigma_rad=sigma_rads[i], sigma_ang=sigma_angs[i]) for i in range(n_cells)]
    network = BVCNetwork(BVCs=BVCs, coeff=1, threshold=0.1)

    # visualise the firing field of the first BVC and the whole place field
    plot_bvc_firing_field(BVCs[0])
    plot_bvc_firing_field(network)

    # distances to boundary: maybe use a truncated Gaussian?
    ds = dist.Normal(loc=3, scale=1).sample(torch.Size([n_data_points]))
    # angles to boundary: sampled uniformly
    phis = dist.uniform.Uniform(low=-torch.pi, high=torch.pi).sample(torch.Size([n_data_points]))

    for i in range(n_data_points):
        output = network.obtain_firing_rate(ds[i], phis[i])
