import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
from utils import *

torch.manual_seed(1234)

class RectangleEnvironment:
    '''
    Rectangular environment to obtain BVC and PC firing rates
    (0,0) is the bottom left corner, north is 0 bearing
    '''
    def __init__(self, l, w):
        self.l = torch.Tensor([l]) # maximum length
        self.w = torch.Tensor([w]) # maximum width
        self.aspect_ratio = l / w
        self.barriers = [] # list of tuples, each sublist (start_loc, end_loc)
        
    def add_straight_barrier(self, start_loc, end_loc):
        '''
        Adds a straight barrier to the environment.
        The barrier is indistinhuishable to the agent
        '''
        x_start, y_start = start_loc
        x_end, y_end = end_loc
        assert x_start >= 0 and x_end <= self.l
        assert y_start >= 0 and y_end <= self.w
        self.barriers.append(torch.Tensor([[x_start + 1e-3, y_start + 1e-3], [x_end, y_end]]))
        
    def compute_wall_dist(self, loc, n_disc=360):
        '''
        Discretises bearing to 360 and compute the distance to each segment. Takes into account any walls and barriers. Vectorised.
        '''
        x, y = loc
        x = torch.Tensor([x]) if not isinstance(x, torch.Tensor) else x
        y = torch.Tensor([y]) if not isinstance(y, torch.Tensor) else y
        assert torch.any(x) > 0 and torch.any(x) < self.l
        assert torch.any(y) > 0 and torch.any(y) < self.w
        try:
            n_mesh_x, n_mesh_y = x.shape
        except:
            n_mesh_x, n_mesh_y = 1, 1
        
        distances = torch.zeros(n_mesh_x, n_mesh_y, n_disc)
        
        angles = torch.ones(n_disc) * 2 *torch.pi / n_disc
        bearings = torch.linspace(0, 2*torch.pi, n_disc)
        
        # compute bearings to 4 corners
        top_right = torch.arctan((self.l-x)/(self.w-y))
        bottom_right = 1/2 * torch.pi + torch.arctan(y/(self.l-x))
        bottom_left = torch.pi + torch.arctan(x/y)
        top_left = 3/2*torch.pi + torch.arctan((self.w-y)/x)

        for bearing in range(n_disc):
            b = torch.Tensor([2 * torch.pi/n_disc * bearing]) + 1e-3 # bearing in rad

            segment_in_top_mask = torch.logical_or(b >= top_left, b < top_right)
            segment_in_right_mask = torch.logical_and(b >= top_right, b < bottom_right)
            segment_in_bottom_mask = torch.logical_and(b >= bottom_right, b < bottom_left)
            segment_in_left_mask = torch.logical_and(b >= bottom_left, b < top_left)

            assert torch.all(segment_in_top_mask*1 + segment_in_right_mask*1 + segment_in_bottom_mask*1 + segment_in_left_mask*1 == torch.ones(n_mesh_x, n_mesh_y))

            distances[:, :, bearing] += torch.multiply(segment_in_top_mask, (self.w-y)/torch.cos(b))
            distances[:, :, bearing] += torch.multiply(segment_in_right_mask, (self.l-x)/torch.cos(b- 1/2*torch.pi))
            distances[:, :, bearing] += torch.multiply(segment_in_bottom_mask, y/torch.cos(b - torch.pi))
            distances[:, :, bearing] += torch.multiply(segment_in_left_mask, x/torch.cos(b- 3/2 * torch.pi))
            
            if len(self.barriers) > 0:
                
                for barrier in self.barriers: # has barrier
                    m1 = 1/torch.tan(b + 1e-3) # (n_mesh_x, n_mesh_y)
                    c1 = y - m1 * x # (n_mesh_x, n_mesh_y)
                    
                    m2 = (barrier[1,1] - barrier[0,1]) / (barrier[1,0] - barrier[0,0]) # m = (y1 - y0) / (x1 - x0)
                    c2 = barrier[1,1] - m2 * barrier[1,0] # c = y1 - m* x1
                    
                    c2 = c2 * torch.ones_like(c1)
                    # system of simultaneuous equations
                    
                    mat_to_inverse = - torch.ones(n_mesh_x, n_mesh_y, 2, 2)
                    mat_to_inverse[:,:,0,0] *= -m1
                    mat_to_inverse[:,:,1,0] *= -m2
                    
                    intersec_loc =  torch.linalg.inv(mat_to_inverse) @  torch.stack((-c1, -c2), dim=-1)[..., None]
                    
                    intersec_x, intersec_y = intersec_loc[:,:,0,0], intersec_loc[:,:,1,0] # (n_mesh_X, n_mesh_y)
                    
                    vec_to_bearing = intersec_loc - torch.stack(loc, dim=-1)[..., None] # vector to bearing
                    barrier_in_direction = torch.sign(vec_to_bearing[:,:,0,0]) == torch.sign(torch.sin(b)) # filter out barriers in negative directions
                    
                    dist_to_barrier = torch.sqrt(torch.pow(vec_to_bearing[:,:,0,0], 2) + torch.pow(vec_to_bearing[:,:,1,0], 2)) # distances to barrier
                    dist_to_barrier = torch.nan_to_num(torch.multiply(1/barrier_in_direction, dist_to_barrier),float('inf'))

                    distances[:,:,bearing] = torch.minimum(distances[:,:,bearing], dist_to_barrier) 
                    
                    
        if n_mesh_x == 1 and n_mesh_y == 1:
            # autoshape
            return distances[0,0,:], bearings, angles
        else:   
            return distances, bearings, angles

    def plot_environment(self):
        '''
        Visualise the environment with all barriers
        '''
        fig, ax = plt.subplots(figsize=(6, 6/self.aspect_ratio))
        background = patches.Rectangle((0, 0), self.w, self.l, facecolor='lightgrey', zorder=-1, edgecolor='grey', linewidth=3)
        ax.add_patch(background)
        
        for barrier in self.barriers:
            ax.plot([barrier[0,0], barrier[1,0]], [barrier[0,1], barrier[1,1]], color='grey', linewidth=3) 
        
        ax.axis('off')
        ax.set_aspect('equal')
        ax.grid(False)
        plt.show()
        
    def generate_mesh(self, n_disc=101):
        '''
        Discritise length and width and generate the distances, bearings, angles from each point.
        '''
        
        x, y = torch.meshgrid(
            torch.linspace(1e-3, self.l.item()-1e-3, n_disc),
            torch.linspace(1e-3, self.w.item()-1e-3, n_disc))
        distances, bearings, angles = self.compute_wall_dist((x,y))
        
        return distances, bearings, angles
    
    def random_sample_locations(self, n_data_points, n_disc=360):
        '''
        Randomly sample n locations in the environment, returns distances, bearings and angles
        
        :returns:
        distances: torch.Tensor shape(n_data_points, n_disc)
        bearing:   torch.Tensor shape(n_data_points, n_disc), discretising (0, 360)
        angles:    torch.Tensor shape(n_data_points, n_disc), one degrees in radian
        '''
        distances, bearings, angles = torch.zeros(n_data_points, n_disc), torch.zeros(n_data_points, n_disc), torch.zeros(n_data_points, n_disc)
        xs = dist.Uniform(low=0, high=self.l).sample(sample_shape=torch.Size([n_data_points]))
        ys = dist.Uniform(low=0, high=self.w).sample(sample_shape=torch.Size([n_data_points]))
        for i, loc in tqdm(enumerate(zip(xs, ys))):
            distance, bearing, angle = self.compute_wall_dist(loc, n_disc=n_disc)
            
            distances[i, :] = distance
            bearings[i, :] = bearing
            angles[i, :] = angle
            
        return distances, bearings, angles
    
    
    def visualise_firing_rates(self, firing_rates, cb=False):
        '''
        Visualises firing map of (predicted) BVCs, PCs or other neurons in the environment
        
        firing_rates: torch.Tensor, shape (mesh_x, mesh_y, #neurons)
        '''
        mesh_x, mesh_y, n_neurons = firing_rates.shape
        
        x, y = torch.meshgrid(
            torch.linspace(1e-3, self.l.item()-1e-3, mesh_x),
            torch.linspace(1e-3, self.w.item()-1e-3, mesh_y))
        
        plt.figure(figsize=(6*n_neurons, 6/self.aspect_ratio))
        for neuron_idx in range(n_neurons):
            plt.subplot(1, n_neurons, neuron_idx+1)
            plt.scatter(x, y, c = firing_rates[:, :, neuron_idx])
            if cb:
                plt.colorbar()
                
            for barrier in self.barriers:
                plt.plot([barrier[0,0], barrier[1,0]], [barrier[0,1], barrier[1,1]], color='grey', linewidth=3) 
        plt.show()
    
    
    def visualise_cell_firing_field(self, neurons, n_disc=101, return_firing_rates=False):
        '''
        Visualises the firing field of a group of neurons
        '''
        
        if not isinstance(neurons, list):
            neurons = [neurons]
        n_neurons = len(neurons)
        
        distances, bearings, angles = self.generate_mesh(n_disc=n_disc)
        
        firing_rates = torch.stack([neuron.compute_firing(distances, bearings, angles) for neuron in neurons], dim=-1)
        self.visualise_firing_rates(firing_rates)
        
        if return_firing_rates:
            return firing_rates
        
class BoundaryVectorCell:
    '''
    A boundary vector cell initiliased with preferred distance (mm) and angle (rad)
    '''
    def __init__(self, d, phi, sigma_zero=122*3, beta=1830, sigma_ang=0.2, multiplier=1, maxfire=False) -> None:
        self.d = torch.Tensor([d])
        self.phi = torch.Tensor([phi])
        self.sigma_zero = sigma_zero                                                 # Hartley 2000: sigZero = 122 mm
        self.beta = beta                                                             # Hartley 2000: beta = 1830 mm
        self.sigma_ang = torch.Tensor([sigma_ang])                                   # Hartley 2000: angSig = 0.2 rad
        self.sigma_rad = torch.Tensor([((self.d/self.beta)+1) * self.sigma_zero])    # Hartley 2000: sigma_rad = [(d/beta) + 1] * sigma_zero
        self.multiplier = multiplier
        self.maxfire = maxfire

    def compute_BVC_firing_single_segment(self, r, theta):
        '''
        Computes the firing of BVC given a current distance and angle to a single segment.
        Firing rate is proportional to product of two gaussians centerred at the preferred distance and angle
        Vectorised: r and theta can be arrays or matrices
        
        :param r:       animal's allocentric distance to boundary
        :param theta:   animal's allocentric bearing to boundary
        '''
        unscaled_firing_rate = torch.exp(-(r - self.d) ** 2 / (2 * self.sigma_rad ** 2))/ \
                torch.sqrt(2 * torch.pi * self.sigma_rad ** 2) * \
                    torch.exp(-(theta - self.phi) ** 2 / (2 * self.sigma_ang ** 2))/ \
                        torch.sqrt(2 * torch.pi * self.sigma_ang ** 2)
        
        return unscaled_firing_rate
    
    def compute_firing(self, distances, bearings, angles):
        '''
        A section of wall at distance r, bearing theta, subtending a angle dtheta at the rat contributes
        dfi = gi(r, theta) * dtheta
        The firing rate is found by integrating to find the net contribution of all the environment's boundaries
        '''
        n_disc = distances.shape[-1]
            
        unscaled_firing_rates = torch.stack([self.compute_BVC_firing_single_segment(distances[... ,i], bearings[..., i]) for i in range(n_disc)], dim=-1)
        firing_rates = self.multiplier * torch.sum(torch.multiply(unscaled_firing_rates, angles), dim=-1)
        
        if self.maxfire is not False:
            return torch.clamp(firing_rates, min=None, max=self.maxfire)
        else:
            return firing_rates


class PlaceCell:
    '''
    A place cell receives input from multiple boundary vector cells
    The firing F_j(x) of PC j at location x is proportional to the thresholded, weighted sum of the N BVC sets that connect to it
    Output is the thresholded firing sum, each at the same allocentric reference frame, scaled by a coefficient A
    '''
    def __init__(self, bvcs=False, connection_weights=False, non_linearity=nn.ReLU(), A=10000, T='auto', max_fire=False):
        
        if bvcs is not False:
            self.bvcs = bvcs
            self.n_bvcs = len(self.bvcs)
            self.connection_weights = connection_weights
            assert len(self.connection_weights) == len(self.bvcs)
            
        self.T = T
        self.non_linearity = nn.ReLU()
        self.A = A
        self.max_fire = max_fire
    
    def create_random_bvcs(self, n_bvcs, max_r=1000):
        self.n_bvcs = n_bvcs
        
        rs = dist.Uniform(low=0, high=max_r).sample(sample_shape=torch.Size([n_bvcs]))
        phis = dist.Uniform(0, 2*torch.pi).sample(sample_shape=torch.Size([n_bvcs]))
        
        self.bvcs = [BoundaryVectorCell(rs[i], phis[i]) for i in range(n_bvcs)]
        self.connection_weights = torch.ones(n_bvcs)
        
    def compute_firing(self, distances, bearings, subtended_angles):
        try:
            self.bvcs
        except:
            raise AttributeError('No BVCs initialised')
            
        bvc_firing_rates = torch.stack([bvc.compute_firing(distances, bearings, subtended_angles) for bvc in self.bvcs], dim=-1)
        weighted_sum = torch.sum(torch.mul(bvc_firing_rates, self.connection_weights), dim=-1)
        
        thresholded_sum = self.A * weighted_sum
        
        if self.T == 'auto':
            self.T = 0.9 * torch.max(thresholded_sum)
            
        thresholded_sum = thresholded_sum - self.T
        
        if self.max_fire is False:
            return self.non_linearity(thresholded_sum)
        else:
            return torch.clamp(self.non_linearity(thresholded_sum), min=None, max=self.max_fire)
    
    # to be implemented
    def BCM_weight_update(self, PC_firing, BVC_firings, D=0.2, F0=0.3, p=3, Phi=lambda F, xi: nn.Tanh(F*(F-xi))):
        '''
        Sustained firing of post-synatic cell below a dynamic threshold leads to weakening of the connection from the pre-synaptic cell
        and vice versa.
        Magnitude of change between PC and BVC i is dw = D(f_i(x) Phi(Fj(x), xi)), xi = (<Fj>/F0)^p*<Fj>
        '''
        Fj = torch.mean(PC_firing)
        xi = torch.pow((Fj/F0), p) * Fj
        dW = [D * BVC_firings[i] *Phi(PC_firing, xi) for i in range(self.n_bvcs)]
        
        return dW

class BVC_PC_network:
    def __init__(self, BVCs:list, n_PCs:int, n_BVCs_per_PC:int, connection_indices=False):
        self.BVCs = BVCs
        self.n_BVCs = len(BVCs)
        self.n_PCs = n_PCs
        self.n_BVCs_per_PC = n_BVCs_per_PC
        
        if not connection_indices: # randomly connect BVCs to PCs
            connection_indices = [random.sample(range(100),self.n_BVCs_per_PC) for PC_idx in range(self.n_PCs)]
        
        self.PCs = [PlaceCell(bvcs=[self.BVCs[i] for i in connection_indices[pc_idx]], 
                                    connection_weights=dist.Normal(1,1).sample(sample_shape=torch.Size([n_BVCs_per_PC])),
                             T='auto') for pc_idx in range(self.n_PCs)]
    
    def compute_population_firing(self, distances, bearings, subtended_angles):
        
        n_locs = distances.shape[0]
        
        BVCs_population_firing = torch.zeros(n_locs, self.n_BVCs)
        PCs_population_firing = torch.zeros(n_locs, self.n_PCs)
        
        for BVC_idx in range(self.n_BVCs):
            BVCs_population_firing[:, BVC_idx] = self.BVCs[BVC_idx].compute_firing(distances, bearings, subtended_angles)
        
        for PC_idx in range(self.n_PCs):
            PCs_population_firing[:, PC_idx] = self.PCs[PC_idx].compute_firing(distances, bearings, subtended_angles)
        
        return BVCs_population_firing, PCs_population_firing