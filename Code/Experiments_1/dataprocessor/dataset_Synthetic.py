#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dataset_Synthetic.py
@Time    :   2022/06/26 17:23:16
@Author  :   QuYue 
@Email   :   quyue1541@gmail.com
@Desc    :   dataset_Synthetic
'''

# %% Import Packages
# Basic
import torch
from typing import Tuple
import pyro
import pyro.distributions as dist

# %% Data Synthetic
def data_synthetic(
    data_number: int = 1000,
    data_dimensions: int = 10,
    ifprint: bool = True
    ) -> Tuple:
    #%%
    # z = pyro.sample('z', dist.Bernoulli(0.5), sample_shape=[10])
    # x = pyro.sample('x', dist.Normal(z, theta1**2*z + theta0**2*(1-z)))
    # t = pyro.sample('t', dist.Bernoulli(0.75*z+0.25*(1-z)))
    # y = pyro.sample('y', dist.Bernoulli(torch.sigmoid(3*(z + 4*t - 2))))

    D = data_dimensions
    N = data_number
    thetaz0 = torch.tensor(1.5)
    thetaz1 = torch.tensor(2.5)
    thetat0 = torch.tensor(1.0)
    thetat1 = torch.tensor(3.0)

    if D == 10:
        wt = torch.tensor([[2.2241,  0.2349, 0.6537,  -2.0293, -0.8639]]).T
        wy = torch.tensor([[1.4137,  3.1837,  0.5843,  1.1793, -2.2991, 
                            -0.1156,  1.6658,  0.3777,-0.7439, -1.0748]]).T
    else:
        wt = pyro.sample('wt', dist.Normal(0, 2), sample_shape=[int(D/2), 1])
        wy = pyro.sample('wy', dist.Normal(0, 1.5), sample_shape=[D, 1])
        wt[-1][0] = 0 - wt[:-1,:].sum()
        wy[-1][0] = 0 - wy[:-1,:].sum()

    z = pyro.sample('z', dist.Bernoulli(0.5), sample_shape=[N, D])
    x = pyro.sample('x', dist.Normal(z, thetaz1**2*z + thetaz0**2*(1-z)))
    t = pyro.sample('t', dist.Bernoulli(torch.sigmoid(z[:,:int(D/2)]@wt)))
    t_0 = torch.zeros_like(t)
    t_1 = torch.ones_like(t)
    y_0 = pyro.sample('y', dist.Normal(3*(z@wy+2*t_0), thetat1**2*t_0 + thetat0**2*(1-t_0)))
    y_1 = pyro.sample('y', dist.Normal(3*(z@wy+2*t_1), thetat1**2*t_1 + thetat0**2*(1-t_1)))
    y_pot = torch.cat([y_0, y_1], dim=1)
    y_obs = (1-t)*y_0 + t*y_1

    if ifprint:
        print(f"Synthetic Dataset Introduction:")
        print(f"z: {z.shape}")
        print(f"x: {x.shape}")
        print(f"t: {t.shape}")
        print(f"y: {y_obs.shape}")
        print(f"y_potential: {y_pot.shape}")

        print(f'Causal effect\t\t{(y_1-y_0).mean().item()}')
        print(f'Statistical effect\t{(y_obs[t==1].mean() - y_obs[t==0].mean()).item()}')
    x = x.numpy()
    y_pot = y_pot.numpy()
    y_obs = y_obs.numpy()
    t = t.numpy()
    # z = z.numpy()

    return {'x': x, 't': t, 'y': y_obs, 'potential_y': y_pot}

# %% Load data
def load(
    data_number: int = 1000,
    data_dimensions: int = 10,
    ifprint: bool = False,
    random_seed: int = None,
    **kwargs
    ) -> Tuple:
    """Load dataset Synthetic.

    Parameters
    ----------
    data_number: int, optional
        Number of samples. (default: 1000)
    data_dimensions: int, optional
        Number of dimensions. (default: 10)
    ifprint: bool, optional
        If print the information of the datasets. (default: False)
    random_seed: int or None, optional
        Random seed.( Default: None )
    kwargs: any
        Other arguments. 

    Returns
    -------
    data: dict{
        x: array or pd.DataFrame
            Features in data.
        t: array or pd.DataFrame
            Treatments in data.
        y: array or pd.DataFrame
            Observed outcomes in data.
        potential_y: array or pd.DataFrame
            Potential outcomes in data.
        }
    """
    # print(random_seed)
    if random_seed is not None:
        pyro.set_rng_seed(random_seed)
    return data_synthetic(data_number, data_dimensions, ifprint)


# %% Main Function
if __name__ == "__main__":
    data = load(data_number=1000, data_dimensions=10, ifprint=True)


