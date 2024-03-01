# Compute alternative deformations using symmetry
# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data\coarseMesh_noBifurcation_5\data2_coarseMesh_noBifurcation_diameter_0.8.pkl'
gr_path = 'data\coarseMesh_noBifurcation_5\graphs_coarseMesh_noBifurcation_diameter_0.8_noBulkNodes.pkl'

# %% open data
with open(data_path, 'rb') as f:
    e_data, n_data, g_data, e_inds, n_inds = pickle.load(f)
for stuff in [e_data, n_data, g_data, e_inds, n_inds]:
    for key in stuff:
        print(f'{key+".shape":30}\t', stuff[key].shape)
    print('')

n_nodes = n_data['pos'].shape[1]
print('n_nodes', n_nodes)

# %% Create all deformation alternatives
bound_eps = 0.01
geom_eps = 0.05
# RVE stays the same
def alter0(pos): return pos.copy()

# RVE shifted to the side
def alter1(pos):
    pos2 = pos.copy()
    pos2[0] += 1.6
    pos2[pos2>=1.6 - 2*bound_eps] = pos2[pos2>=1.6 - 2*bound_eps] - 3.2
    return pos2

# RVE shifted up/down 
def alter2(pos):
    pos2 = pos.copy()
    pos2[1] += 1.6
    pos2[pos2>=1.6 - 2*bound_eps] = pos2[pos2>=1.6 - 2*bound_eps] - 3.2
    return pos2

# RVE shifted to the side AND up/down
def alter3(pos):
    pos2 = pos.copy()
    pos2 += 1.6
    pos2[pos2>=1.6 - 2*bound_eps] = pos2[pos2>=1.6 - 2*bound_eps] - 3.2
    return pos2

# compute all alternative deformations
inds = np.empty((n_nodes, 8), dtype=int)
pos_final2 = np.empty((*n_data['pos_affine'].shape, 8))
w = n_data['pos_final'] - n_data['pos_affine']
mean_pos = np.mean(n_data['pos_final'], axis=-1, keepdims=True)
for i, upsidedown in enumerate([False , True]):
    for j, alter in enumerate([alter0, alter1, alter2, alter3]):
        # get new alternative
        pos2 = alter(n_data['pos'])
        if upsidedown:
            pos2 = -pos2
            pos2[pos2>=1.6 - 2*bound_eps] = pos2[pos2>=1.6 - 2*bound_eps] - 3.2

        # create index for each node that maps the correct deformation of another node to it
        inds_temp = []
        for k in range(n_nodes):
            bools = np.abs(n_data['pos'][:, k].reshape(2, 1) - pos2) < geom_eps
            ind_temp = np.where(np.prod(bools, axis=0))
            if len(ind_temp) == 1 and len(ind_temp[0]) == 1:
                inds_temp.append(ind_temp[0][0])
            else:
                raise ValueError(f'There should be 1 index per node, instead {len(ind_temp[0])} were found for alternative {i}, upsidedown {upsidedown}, node {k}')

        print(i, j, 4*i+j)
        if upsidedown:
            pos_final2[:, :, :, 4*i+j] = n_data['pos_affine'] - w[:, :, inds_temp]
        else:
            pos_final2[:, :, :, 4*i+j] = n_data['pos_affine'] + w[:, :, inds_temp]

        # # fix mean pos back into place
        mean_pos_temp = np.mean(pos_final2[..., 4*i+j], axis=-1, keepdims=True)
        pos_final2[:, :, :, 4*i+j] += mean_pos - mean_pos_temp

        inds[:, 4*i+j] = inds_temp
            
#%% Check if some alternatives are always the same
ignore_upsidedown = True

if ignore_upsidedown:
    n_alts = 4 
    labels = ['original', 
            'shifted right', 
            'shifted up', 
            'shifted right and up'
            ]
    bools = np.empty((len(pos_final2), 6), dtype=bool)
else:
    n_alts = 8
    labels = ['original', 
            'shifted right', 
            'shifted up', 
            'shifted right and up', 
            'original, upside down',
            'shifted right, upside down',
            'shifted up, upside down',
            'shifted right and up, upside down'
            ]
    bools = np.empty((len(pos_final2), 28), dtype=bool)
print('n_alts', n_alts)

# %% try to find correct tolerance
# compare all alternatives to each other (using np.isclose())
for atol in 10**np.linspace(-3, 0, 20):
    # iterate over all loadcases
    for k, pos_temp in enumerate(pos_final2):
        # iterate over all combinations of alternatives
        m = 0
        for i in range(n_alts-1):
                for j in range(i+1, n_alts):
                    bools[k][m] = np.isclose(pos_temp[..., i], pos_temp[..., j], rtol=0, atol=atol).all()
                    m += 1

    # print(f'{np.all(bools, axis=0)}')
    # print(np.unique(np.sum(bools, axis=1), return_counts=True))
    # check which options are always the same
    m = 0
    all_results = np.all(bools, axis=0)
    # print('===============================================')
    # print('Always the same:')
    # iterate over all combinations of alts
    for i in range(n_alts-1):
        for j in range(i+1, n_alts):
            if all_results[m]:
                # print(f'{i} {j} {labels[i]:28} {labels[j]:34}')  #{all_results[m]}')
                pass
            m += 1
    combis, examples, inv, counts = np.unique(bools, axis=0, return_counts=True, return_inverse=True, return_index=True)
    # print('bools.shape', bools.shape)
    # print('combis.shape', combis.shape)

    print('\n===============================================')
    print(f'{atol:5} {len(combis)}')
    print(f'{"counts":30}', end=' ')
    for c in counts:
        print(f'{c:6}', end=' ')
    print(f'\n{"example index":30}', end=' ')
    for ex in examples:
        print(f'{ex:6}', end=' ')


# %%
# iterate over all loadcases
for k, pos_temp in enumerate(pos_final2):
    # iterate over all combinations of alternatives
    m = 0
    for i in range(n_alts-1):
            for j in range(i+1, n_alts):
                bools[k][m] = np.isclose(pos_temp[..., i], pos_temp[..., j], rtol=0, atol=0.15).all()
                m += 1

# print(f'{np.all(bools, axis=0)}')
# print(np.unique(np.sum(bools, axis=1), return_counts=True))
# check which options are always the same
m = 0
all_results = np.all(bools, axis=0)
print('===============================================')
print('Always the same:')
# iterate over all combinations of alts
for i in range(n_alts-1):
    for j in range(i+1, n_alts):
        if all_results[m]:
            # print(f'{i} {j} {labels[i]:28} {labels[j]:34}')  #{all_results[m]}')
            pass
        m += 1
combis, examples, inv, counts = np.unique(bools, axis=0, return_counts=True, return_inverse=True, return_index=True)
print('bools.shape', bools.shape)
print('combis.shape', combis.shape)
print(f'{atol:5} {len(combis)}')

# bools[0, 1]: whether every node in graph 0, has a corresponding node with the same position doing comparison 1 of alts (which would be 'original' with 'shifted right'). So if bools[0, 1] = True, then 'original' and 'shifted up' are the same configuration for graph 0.
# check which combinations of alternatives there are with np.unique: len of combis is the nr of symmetry types in the dataset
# each column of bools is one 'set' of correspondences that occurs

print('===============================================')
print('Comparisons between alternatives:')
m = 0
# iterate over all combinations of alts
for i in range(n_alts-1):
    for j in range(i+1, n_alts):
        print(f'{i} {j} {labels[i]:28} {labels[j]:34}', end=' ')
        for comb in combis[:, m]:
            print(f'{str(comb):6}', end=' ')
        print('')
        m += 1

print('===============================================')
print(f'{"counts":66}', end=' ')
for c in counts:
    print(f'{c:6}', end=' ')
print(f'\n{"example index":66}', end=' ')
for ex in examples:
    print(f'{ex:6}', end=' ')

# %%
# # turning upside down turns out to be unnecessary, so select only the 1st 4 alternatives
# pos_final2 = pos_final2[..., :4]
# pos_final2.shape

#%% Plot to check on alternative deformations
ind = 46     
%matplotlib qt
# plt.scatter(*pos)
for i in range(n_alts):  # [0, 2]:   # [0]:  # 
    # plot initial positions of all alternatives
    # plt.scatter(*pos_affine[-1][:, inds[:, i]], label=i, s=3)

    # plot affine positions of all alternatives
    # plt.scatter(*pos_affine[-1][:, inds[:, i]], label=i, s=3, alpha=0.3)

    # plot affine positions using affine deformation
    # plt.scatter(*(pos[:, inds[:, i]] + U_affine[-1][:, inds[:, i]]), label=i, s=3, alpha=0.3)

    # plot final positions using U_affine and w
    # plt.scatter(*(pos
    #         + U_affine[-1]
    #         + w[-1][:, inds[:, i]]
    #     ), label=i, s=3, alpha=0.3)
    
    # plot final positions of all alternatives

    # plot edges
    x, y = pos_final2[ind, ..., i].T[e_data['edge_index'][:, e_inds['hole_boundary']]].T
    x, y = x.T, y.T
    print(x.shape)
    bools = (np.abs(x[0] - x[1]) < 1.6)*(np.abs(y[0] - y[1]) < 1.6)
    plt.plot(x[:, bools], y[:, bools], c='tab:blue', alpha=0.3, zorder=-1)
    plt.scatter(*(pos_final2[ind, :, :, i]), s=5, label=labels[i], alpha=0.5)

    pass

# plt.scatter(*pos_final[-1], alpha=0.3, s=3)
# plt.scatter(*(np.dot(F[-1], pos + 1.6) + w[-1]), alpha=0.3, s=3)

plt.gca().set_aspect('equal')
plt.legend()
plt.grid()
# %%
# add symmetry type to graphs
with open(gr_path, 'rb') as f:
    data_list0 = pickle.load(f)

for graph, symm in zip(data_list0, inv):
    graph.symm_type = symm

import os
gr_path2 = os.path.splitext(gr_path)[0] + '_symmtype.pkl'
with open(gr_path2, 'wb') as f:
    pickle.dump(data_list0, f)
# %%
