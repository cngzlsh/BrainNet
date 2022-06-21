import torch
import torch.nn as nn
import torch.distributions as dist
from utils import plot_bvc_firing_field

torch.manual_seed(1234)

class RectangleEnvironment:
    '''
    Rectangular environment to obtain BVC and PC firing rates
    '''
    def __init__(self, l, w):
        self.l = torch.Tensor([l]) # maximum length
        self.w = torch.Tensor([w]) # maximum width
        
    def compute_wall_dist(self, loc):
        '''
        Calculates the distances, bearings and subtended angles of a rat location
        returns distances, bearings, angles. Each list of length 4
        left/up - positive dist, right/down - negative dist
        '''
        x, y = loc
        x = torch.Tensor([x]) if not isinstance(x, torch.Tensor) else x
        y = torch.Tensor([y]) if not isinstance(y, torch.Tensor) else y
        assert torch.any(x) > 0 and torch.any(x) < self.l
        assert torch.any(y) > 0 and torch.any(y) < self.w
        
        distances, bearings, angles = [], [], []
        
        # up wall
        distances.append(y)
        bearings.append(0 * torch.pi)
        angles.append(torch.arctan(x/y) + torch.arctan((self.l-x)/y))

        # right wall
        distances.append(self.l - x)
        bearings.append(1/2 * torch.pi)
        angles.append(torch.arctan(y/(self.l-x)) + torch.arctan((self.w-y)/(self.l-x)))
        
        # down wall
        distances.append(self.w - y)
        bearings.append(1 * torch.pi)
        angles.append(torch.arctan(x/(self.w-y)) + torch.arctan((self.l-x)/(self.w-y)))

        # left wall
        distances.append(x)
        bearings.append(3/2* torch.pi)
        angles.append(torch.arctan(y/x) + torch.arctan((self.w-y)/x))
        
        return distances, bearings, angles


class BVC:
    '''
    A boundary vector cell with preferred distance and angle
    Hartley 2000: sigma_ang = 0.2 fixed, sigma_rad = [(peak_dist/beta)+1]*sigma_zero where sigma_zerp speifies sigma for zero distances and
    beta controls the rate at which sigma_rad increases
    Defaults - all distances in mm
    '''
    def __init__(self, r, theta, sigma_zero=122, beta=1830, sigma_ang=0.2, scaling_factor=1) -> None:
        self.r = torch.Tensor([r])
        self.theta = torch.Tensor([theta])
        self.sigma_zero = sigma_zero                        # Hartley 2000: sigZero = 122 mm
        self.beta = beta                                    # Hartley 2000: beta = 1830 mm
        self.sigma_ang = torch.Tensor([sigma_ang])          # Hartley 2000: angSig = 0.2 rad
        self.sigma_rad = torch.Tensor([((self.r/self.beta)+1) * self.sigma_zero])
        self.scaling_factor = scaling_factor

    def obtain_firing_rate_single_boundary(self, d, phi):
        '''
        Computes the firing of BVC given a current distance and angle to a single boundary.
        Firing rate is proportional to product of two gaussians centerred at the preferred distance and angle
        Vectorised: d and phi can be arrays or matricesd
        '''
        unscaled_firing_rate = torch.exp(-(self.r - d) ** 2 / (2 * self.sigma_rad ** 2))/ \
                torch.sqrt(2 * torch.pi * self.sigma_rad ** 2) * \
                    torch.exp(-(self.theta - phi) ** 2 / (2 * self.sigma_ang ** 2))/ \
                        torch.sqrt(2 * torch.pi * self.sigma_ang ** 2)
        
        return unscaled_firing_rate
    
    def obtain_net_firing_rate(self, distances, bearings, subtended_angles):
        '''
        A section of wall at distance r, bearing theta, subtending a angle dtheta at the rat contributes
        dfi = gi(r, theta) * dtheta
        The firing rate is found by integrating to find the net contribution of all the environment's boundaries
        '''
        n_boundaries = len(distances)
        net_unscaled_firing_rates = torch.stack([self.obtain_firing_rate_single_boundary(distances[i], bearings[i]) for i in range(n_boundaries)], dim=0)
        subtended_angles = torch.stack(subtended_angles, dim=0)
        return self.scaling_factor * torch.sum(torch.multiply(net_unscaled_firing_rates, subtended_angles), dim=0)
        

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
    l = 120
    w = 80
    env = RectangleEnvironment(l, w)
    bvc1 = BVC(150, 0)
    bvc2 = BVC(100, 1/2*torch.pi)
    bvc3 = BVC(120, 1/3*torch.pi)
    plot_bvc_firing_field(bvc2)

    n = 100
    x, y = torch.meshgrid(torch.linspace(1e-5, l-1e-5, n), torch.linspace(1e-5, w-1e-5, n))
    distances, bearings, angles = env.compute_wall_dist((x,y))
    fr1 = bvc1.obtain_net_firing_rate(distances, bearings, angles)