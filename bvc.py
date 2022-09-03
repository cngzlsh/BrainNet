import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from utils import *
from tqdm import tqdm

torch.manual_seed(1234)
random.seed(1234)

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
        self.barriers.append(torch.Tensor([[x_start + 1e-4, y_start + 1e-4], [x_end, y_end]]))
        
    def remove_all_barriers(self):
        '''
        Removes all barriers in the environment.
        '''
        self.barriers = []
        
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
            b = torch.Tensor([2 * torch.pi/n_disc * bearing])
            # bearing in rad

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
                    m1 = 1/(torch.tan(b)+1e-3) # (n_mesh_x, n_mesh_y)
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
                    
                    barrier_in_direction = torch.logical_and(
                        torch.logical_and(
                            torch.logical_xor(intersec_x <= barrier[0,0], intersec_x <= barrier[1,0]),
                            torch.logical_xor(intersec_y <= barrier[0,1], intersec_y <= barrier[1,1])),
                        torch.sign(vec_to_bearing[:,:,0,0]) == torch.sign(torch.sin(b))) # filter out barriers in negative directions
                    
                    dist_to_barrier = torch.sqrt(torch.pow(vec_to_bearing[:,:,0,0], 2) + torch.pow(vec_to_bearing[:,:,1,0], 2)) # distances to barrier
                    dist_to_barrier = torch.nan_to_num(torch.multiply(1/barrier_in_direction, dist_to_barrier),float('inf'))

                    distances[:,:,bearing] = torch.minimum(distances[:,:,bearing], dist_to_barrier) 
                    
                    
        if n_mesh_x == 1 and n_mesh_y == 1:
            # autoshape
            return distances[0,0,:], bearings, angles
        else:   
            return distances, bearings, angles
        
        
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
    
    
    def visualise_firing_rates(self, firing_rates, cb=False, fname=False):
        '''
        Visualises firing map of (predicted) BVCs, PCs or other neurons in the environment
        
        firing_rates: torch.Tensor, shape (mesh_x, mesh_y, #neurons)
        '''
        mesh_x, mesh_y, n_neurons = firing_rates.shape
        
        x, y = torch.meshgrid(
            torch.linspace(0, self.l.item(), mesh_x),
            torch.linspace(0, self.w.item(), mesh_y))
        
        plt.figure(figsize=(6*n_neurons, 6/self.aspect_ratio), frameon=False)
        for neuron_idx in range(n_neurons):
            plt.subplot(1, n_neurons, neuron_idx+1)
            plt.scatter(x, y, c = firing_rates[:, :, neuron_idx])
            plt.xticks([])
            plt.yticks([])
            if cb:
                plt.colorbar()
                
            for barrier in self.barriers:
                plt.plot([barrier[0,0], barrier[1,0]], [barrier[0,1], barrier[1,1]], color='grey', linewidth=3) 
        
        if fname is not False:
            plt.savefig('./figures/' + fname, dpi=350, bbox_inches='tight')
        plt.show()
    
    
    def visualise_cell_firing_field(self, neurons, n_disc=101, return_firing_rates=False, cb=False, fname=False):
        '''
        Visualises the firing field of a group of neurons
        '''
        
        if not isinstance(neurons, list):
            neurons = [neurons]
        n_neurons = len(neurons)
        
        distances, bearings, angles = self.generate_mesh(n_disc=n_disc)
        
        firing_rates = torch.stack([neuron.compute_firing(distances, bearings, angles) for neuron in neurons], dim=-1)
        self.visualise_firing_rates(firing_rates, cb=cb, fname=fname)
        
        if return_firing_rates:
            return firing_rates
    
    def check_new_loc(self, current_loc, proposed_loc):
        '''
        Returns False if there is an intersection or is outside boundary
        '''
        
        if proposed_loc[0] > self.l or proposed_loc[0] < 0:
            return False
        if proposed_loc[1] > self.w or proposed_loc[1] < 0:
            return False
        
        for barrier in self.barriers:
            m1 = (proposed_loc[1] - current_loc[1]) / (proposed_loc[0] - current_loc[0] + 1e-3)
            c1 = proposed_loc[1] - m1 * proposed_loc[0]
            
            m2 = (barrier[1,1] - barrier[0,1]) / (barrier[1,0] - barrier[0,0] + 1e-3) # m = (y1 - y0) / (x1 - x0)
            c2 = barrier[1,1] - m2 * barrier[1,0] # c = y1 - m* x1
            
            intersec_loc = torch.linalg.inv(torch.Tensor([[m1, -1], [m2, -1]])) @ torch.Tensor([[-c1], [-c2]])
            
            criterion1 = torch.logical_xor(intersec_loc[0] <= proposed_loc[0], intersec_loc[0] <= current_loc[0])
            criterion2 = torch.logical_xor(intersec_loc[1] <= proposed_loc[1], intersec_loc[1] <= current_loc[1])
            criterion3 = torch.logical_xor(intersec_loc[0] <= barrier[0,0], intersec_loc[0] <= barrier[1,0])
            criterion4 = torch.logical_xor(intersec_loc[1] <= barrier[0,1], intersec_loc[1] <= barrier[1,1])
            if torch.all(torch.vstack([criterion1, criterion2, criterion3, criterion4])):
                return False
        return True
    
    def plot_environment_and_trajectory(self, loc_history=False, axis=False, fname=False):
        '''
        Plots the environment with barriers inserted
        If trajectory is provided, also plots the trajectory
        '''
        fig, ax = plt.subplots(figsize=(6, 6/self.aspect_ratio))
        background = patches.Rectangle((0, 0), self.l, self.w, edgecolor='grey', facecolor='lightgrey', zorder=-1, fill=True, lw=5)
        ax.add_patch(background)
        
        if self.barriers is not []:
            for barrier in self.barriers:
                ax.plot([barrier[0,0], barrier[1,0]], [barrier[0,1], barrier[1,1]], color='grey', linewidth=3) 
        plt.xlim([-1, self.l+1])
        plt.ylim([-1, self.w+1])
        if not axis:
            ax.axis('off')
        ax.set_aspect('equal')
        ax.grid(False)
        
        if loc_history is not False:
            ax.scatter(loc_history[:,0], loc_history[:,1], c=np.arange(loc_history.shape[0]), marker='o', s=1)

        if fname is not False:
            plt.savefig('./figures/' + fname, dpi=350, bbox_inches='tight')
        plt.show()


class CircleEnvironment:
    '''
    Circular environment to obtain BVC and PC firing rates
    '''
    def __init__(self, radius):
        self.radius = torch.Tensor([radius])
    
    def compute_wall_dist(self, loc, n_disc=360):
        x, y = loc
        x = torch.Tensor([x]) if not isinstance(x, torch.Tensor) else x
        y = torch.Tensor([y]) if not isinstance(y, torch.Tensor) else y
        assert torch.any(torch.pow(x, 2) + torch.pow(y, 2)) < torch.any(torch.pow(self.radius, 2)), 'location not in environment'
        try:
            n_mesh_x, n_mesh_y = x.shape
        except:
            n_mesh_x, n_mesh_y = 1, 1
        
        angle3 = torch.arctan(x/y) # [n_mesh_x, n_mesh_y]
        dist_to_centre = torch.pow(x, 2) + torch.pow(y, 2)
        
        pass
        
    def generate_mesh(self, n_disc=101):
        pass
    
    def random_sample_locations(self, n_data_points, n_disc=360):
        pass
    
    def visualise_firing_rates(self, firing_rates, cb=False):
        pass
        
    def visualise_cell_firing_field(self, neurons, n_disc=101):
        pass


class Agent:
    
    '''
    An artificial agent that roams in an environment. Must be passed with an environment class.
    
    params:
    initial_v:        initial velocity, (2,) torch Tensor
    initial_pos:      initial position, (2,) torch Tensor
    dt:               discretised time intercal
    bounce_param:     concentration parameter of random rotation when agent hits a wall
    drift:            stochastic drift in velocity, defaults to 0, can be a lambda function
    diffusion:        diffusion coefficient function, variations in agent's velocity each time step
    '''
    def __init__(self, env, initial_v=False, initial_pos=False, dt=0.1, bounce_param=2.0, drift=0.0, diffusion=1.0):
        
        self.env = env
        self.dt = dt
        self.bounce_param = bounce_param
        self.drift = drift
        self.diffusion = diffusion
        self.reset(initial_pos, initial_v)
    
    def reset(self, initial_pos=False, initial_v=False):
        '''
        Resets time step, initial positions and velocity
        '''
        if initial_pos:
            self.loc = initial_pos
            assert self.loc[0] >= 0 and self.loc[0] <= self.env.l
            assert self.loc[1] >= 0 and self.loc[1] <= self.env.w
            
        else:
            x = dist.Uniform(low=0, high=self.env.l).sample(sample_shape=torch.Size([1]))
            y = dist.Uniform(low=0, high=self.env.w).sample(sample_shape=torch.Size([1]))
            self.loc = torch.concat((x,y))[:,0]
            
        if initial_v:
            self.v = initial_v
        else:
            self.v = dist.Normal(0,2).sample(sample_shape=torch.Size([2]))
            
        self.loc_history = [self.loc]
        self.t = 0
    
    
    def step(self):
        '''
        One time interval movement
        '''
        self.t += self.dt
        current_loc = self.loc
        proposed_loc = self.loc + self.dt * self.v
        
        if self.env.check_new_loc(current_loc, proposed_loc):
            self.loc = proposed_loc
            self.loc_history.append(self.loc)
        else:
            theta = dist.von_mises.VonMises(0,self.bounce_param).sample() # rotation angle, drawn from circular Gaussian
            rotation = torch.Tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
            self.v = - self.v @ rotation
            
        self.v += diffusion_process(self.v, self.dt, self.t, self.drift, self.diffusion)
    
    def run(self, time):
        '''
        Generate history of the agent's position in the environment up to a certain time
        '''
        while self.t < time:
            self.step()
            
        return torch.vstack(self.loc_history)


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
            self.T = 0.8 * torch.max(thresholded_sum)
            
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
    '''
    BVC-PC network.
    If connection weights are not given, they are drawn randomly and connection weights are sampled ~ N(1,1)
    '''
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


def plot_bvc_firing_field(bvcs, max_d='auto', axis='on', n=200, fname=False):
    '''
    Plots firing field of (multiple) BVCs
    '''
    if not isinstance(bvcs, list):
        bvcs = [bvcs]
    n_bvcs = len(bvcs)
    
    if max_d =='auto':
        max_d = int(max([i.d for i in bvcs]) * 1.5)
    rads = torch.linspace(0, 2*torch.pi, n)
    ds = torch.linspace(0, max_d, n)

    rads_mat, ds_mat = torch.meshgrid(rads, ds)

    plt.figure(figsize=(4*n_bvcs, 4))
    for i in range(n_bvcs):
        ax = plt.subplot(1, n_bvcs,i+1, projection='polar')
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N', offset=0)
        if axis == 'off':
            ax.set_xticks([])
            ax.set_yticks([])
        firing_rates = bvcs[i].compute_BVC_firing_single_segment(ds_mat, rads_mat)
        ax.scatter(rads_mat, ds_mat, c=firing_rates, s=1, cmap='hsv', alpha=0.75)
    
    if fname is not False:
            plt.savefig('./figures/' + fname, dpi=350, bbox_inches='tight')
    plt.show()


def SSIM(x, y, k1=0.01, k2=0.03, alpha=1, beta=1, gamma=1):
    '''
    Computes the Structural Similarity Index (SSIM) of two images x and y
    '''
    assert x.shape == y.shape
    
    L = torch.max(torch.max(x), torch.max(y))
    
    x, y = x.flatten(), y.flatten()
    
    mu_x, mu_y = torch.mean(x), torch.mean(y)
    sigma_x, sigma_y = torch.std(x, unbiased=True), torch.std(y, unbiased=True)
    sigma_xy = torch.cov(torch.stack((x,y)))[0,1]
    
    c1 = (k1 * L) ** 2
    luminance = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y **2 + c1)
    
    c2 = (k2 * L) ** 2
    contrast = (2 * sigma_x * sigma_y + c2) / (sigma_x **2 + sigma_y **2 + c2)
    
    c3 = c2 / 2
    structure = (sigma_xy + c3) / (sigma_x * sigma_y + c3)
    
    SSIM = torch.pow(luminance, alpha) * torch.pow(contrast, beta) * torch.pow(structure, gamma)
    
    return SSIM