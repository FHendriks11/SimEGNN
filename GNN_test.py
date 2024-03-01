# %%
# import stuff
import pickle
import sys
import os
import importlib
import gc

import torch
import matplotlib.pyplot as plt
import numpy as np
import funcs_helpers as fh
import mlflow
import torch_geometric as tg

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

'''All test cases:

test data of original data set
test data rotated, reflected, (translated: too boring?)
test data scaled
test data shifted RVE
test data extended RVE (16 holes instead of 4)

more interesting ones:
(no hole case - should reproduce constitutive model: not possible without bulk nodes)
different sized holes
different mesh (martins mesh)
finer mesh
'''

# %%
# load data
data_path0 = r"data\coarseMesh_noBifurcation_5\graphs_coarseMesh_noBifurcation_diameter_0.9_5_noBulkNodes_2.pkl"
data_path1 = r"data\coarseMesh_noBifurcation_5\graphs_coarseMesh_noBifurcation_diameter_0.9_5_noBulkNodes_largerRVE.pkl"

data = {}
with open(data_path0, 'rb') as f:
    data['reference'] = pickle.load(f)
print(data['reference'][0])
print(len(data['reference']))

with open(data_path1, 'rb') as f:
    data['extended RVE'] = pickle.load(f)
print(data['extended RVE'][0])
print(len(data['extended RVE']))

# with open(data_path2, 'rb') as f:
#     data['finer mesh'] = pickle.load(f)
# print(data['finer mesh'][0])
# print(len(data['finer mesh']))

# # %%
# with open(data_path3, 'rb') as f:
#     temp = pickle.load(f)
# data['diameter 0.8 (buckled)'] = [graph for graph in temp if graph.symm_type != 4]
# data['diameter 0.8 (unbuckled)'] = [graph for graph in temp if graph.symm_type == 4]
# print(data['diameter 0.8 (buckled)'][0])
# print(len(data['diameter 0.8 (buckled)']))
# print(len(data['diameter 0.8 (unbuckled)']))

# to do: martin's mesh

# %%
for key in data: print(key, len(data[key]))

# %%
# only keep validation data
trajs = [graph.traj.numpy() for graph in data['reference']]
train_inds, val_inds = fh.split_trajs(trajs, 2, split_sizes=(0.7, 0.3))

for key in ['reference', 'extended RVE']:
    data[key] = [data[key][ind] for ind in val_inds]

# %%
# rotate/reflect/scale
scale_factor = 1.5
R_arr = [torch.tensor([[0.707107, -0.707107],[0.707107, 0.707107]]),  # rot
         torch.tensor([[-1.0, -0],[0, 1.0]]),  # refl
         torch.tensor([[scale_factor, 0],[0, scale_factor]])]   # scale
keys = ['rotated', 'reflected', 'scaled']
for R, key in zip(R_arr, keys):
    data[key] = []
    err = []
    for graph in data['reference']:
        graph = graph.clone()

        R = R.to(graph.y.device)
        # print(R.device)
        # print(graph.y.device)
        graph.y = torch.matmul(R, graph.y.T).T
        graph.pos = torch.matmul(R, graph.pos.T).T
        graph.r = torch.matmul(R, graph.r.T).T
        # graph.mean_pos = torch.matmul(R, graph.mean_pos.T).T
        if key == 'scaled':
            graph.d = scale_factor*graph.d
        if key != 'scaled':
            graph.P = torch.einsum('lj,ijk,km->ilm', R, graph.P, R.T)
            graph.D = torch.einsum('nj,ok,pl,qm,ijklm->inopq', R, R, R, R, graph.D)
            graph.F = torch.einsum('ij,lk,mjk->mil', R, R, graph.F)

        graph.mean_pos = torch.mean(graph.y, axis=0, keepdim=True)
        # mean_pos2 = torch.mean(graph.y, axis=0, keepdim=True)
        # err.append((mean_pos2 - graph.mean_pos).cpu().numpy())
        data[key].append(graph)
    # print(key, 'MAE in mean_pos:', np.mean(np.abs(err)))

# %%
# shifted RVE
# vector describing how the RVE is shifting
shift_vec = torch.tensor([0.8, 0.8])

# basis vectors spanning the RVE
basis_vecs = torch.tensor([[3.2, 0], [0, 3.2]])  #.to(data['reference'][0].y.device)
data['shifted RVE'] = []
for graph in data['reference']:
    graph = graph.clone()
    basis_vecs = basis_vecs.to(graph.y.device)
    shift_vec = shift_vec.to(graph.y.device)
    graph.pos += shift_vec
    graph.y += torch.matmul(graph.F, shift_vec)

    bools1 = graph.pos[:, 0] > 1.6
    graph.pos[bools1] -= basis_vecs[0]
    graph.y[bools1] -= torch.matmul(graph.F, basis_vecs[0])

    bools2 = graph.pos[:, 1] > 1.6
    graph.pos[bools2] -= basis_vecs[1]
    graph.y[bools2] -= torch.matmul(graph.F, basis_vecs[1])

    graph.mean_pos = torch.mean(graph.y, dim=0, keepdim=True)
    data['shifted RVE'].append(graph)

# %% noisy distances (to check how errors grow)
data['noisy distances'] = []
for graph in data['reference']:
    graph = graph.clone()
    graph.d += torch.randn(*graph.d.shape)*1e-7
    data['noisy distances'].append(graph)

data['noisy d and r'] = []
for graph in data['reference']:
    graph = graph.clone()
    graph.d += torch.randn(*graph.d.shape)*1e-7
    graph.r += torch.randn(*graph.r.shape)*1e-7
    data['noisy d and r'].append(graph)



# %%
# plot 2

# plot all nodes in original location
# plt.scatter(*(n_data['pos']), c='black', s=1)  #, s=50, alpha=0.5)

key = 'extended RVE'  # 'finer mesh'  #
n_nodes = data['reference'][0]['y'].shape[0]
# plot nodes on hole boundary
plt.scatter(*data[key][1]['y'][:n_nodes].T, s=10)  #, marker='x', s=50, c=quad)

# plot edges
pos1 = data[key][0].y.cpu().numpy()[data[key][0].edge_index.cpu().numpy()[0]]
pos2 = pos1 + data[key][0].r.cpu().numpy()
x = np.stack((pos1[:, 0], pos2[:, 0]))
y = np.stack((pos1[:, 1], pos2[:, 1]))
plt.plot(x, y, c='red', alpha=0.3, zorder=-1)

# make plot pretty
plt.gca().set_aspect('equal')
plt.grid()

# %% Check std
print(f'{"test case":17} {"Mean":>11} {"Std":>11}')
for var in ['y', 'W', 'P', 'D']:
    print('====================', var, '====================')
    for key in data:
        temp = [getattr(graph, var).detach().cpu().numpy() for graph in data[key]]
        print(f'{key:17} {np.mean(temp):11.5} {np.std(temp):11.5}')

# %%
for var in ['W', 'P', 'D']:
    print('====================', var, '====================')
    for key in ['reference', 'rotated', 'reflected']:
        print(key)
        print(getattr(data[key][0], var))



# %%
#test models
results_MSE = {}
results_R2 = {}
results_frob = {}
client = mlflow.tracking.MlflowClient()
bs = 64

run_ids = ['9453aa73b9ee4b42ac9560ff37693d6f',
           'f25933db5e5545388265e2e9261edca3',
           '64e45fceb1eb46b6a977851c7123bca8',
           '0f54a8a568094a3085cc387fe21c3d38',
           'bdca87c393db4da3b387a805426d25d2',
           'a49f45e0cc3743ab914283c4ea0e6d60',
           'f09812771fec478eb1195216f3ae018d',
           ]
model_names = ['GNN',
         'GNN, DA ×1',
         'GNN, DA ×2',
         'EGNN',
         'EGNN, DA ×1',
         'EGNN, DA ×2',
         'SimEGNN',
        #  'EGNNmod2_bigger'
        ]
for run_id, name in zip(run_ids, model_names):
    print('================================================')
    print('testing model', name)
    torch.cuda.empty_cache()
    gc.collect()

    results_MSE[name] = {}
    results_R2[name] = {}
    results_frob[name] = {}
    # import model

    art_path = client.get_run(run_id).info.artifact_uri[8:]
    print(art_path)

    # Fetch the logged artifacts
    artifacts = client.list_artifacts(run_id)

    # find the files with the weights of the model and the parameters needed to initialize it
    for artifact in artifacts:
        if 'weights.pt' in artifact.path:
            model_path = os.path.join(art_path, artifact.path)
            print(model_path)
        elif artifact.path.endswith('model_init_params.pkl'):
            init_params_path = os.path.join(art_path, artifact.path)
            print(init_params_path)

    # import initialization parameters
    with open(init_params_path, 'rb') as f:
        init_params = pickle.load(f)

    # make sure model definition is imported from the right directory
    sys.path.insert(0, art_path)

    # import model definition
    files = os.listdir(art_path)
    fleur_GNN_definition = [file for file in files if file.startswith('fleur_GNN')]
    if len(fleur_GNN_definition) > 1:
        raise Exception('Multiple fleur_GNN definitions found')
    elif len(fleur_GNN_definition) == 0:
        raise Exception('No fleur_GNN definition found')

    fG = importlib.import_module(fleur_GNN_definition[0].split('.')[0])
    fG = importlib.reload(fG)

    # create model, load params
    model = fG.MyGNN(**init_params)
    model.load_state_dict(torch.load(model_path))
    print(model)

    scaling_factors = eval(mlflow.get_run(run_id).data.params['scaling_factors'])
    print('scaling_factors:', scaling_factors)
    device = torch.device('cuda')
    model.to(device)

    for key, value in data.items():
        print(key)

        # largerRVE graphs are 4 times larger, so need more memory, so smaller batch size needed
        if (key == 'extended RVE') or (key == 'finer mesh'): bs2 = bs//4
        else: bs2 = bs

        val_loader = tg.loader.DataLoader(value, batch_size=bs2)
        n_graphs = len(value)

        MSEs = np.zeros(4)

        reals = {key: [] for key in ['y', 'W', 'P', 'D']}
        preds = {key: [] for key in ['y', 'W', 'P', 'D']}
        for i, batch in enumerate(val_loader):

            batch.to(device)

            with torch.no_grad():
                preds_temp = model(batch)

            for j, var in enumerate(['y', 'W', 'P', 'D']):
                if var == 'y':
                    # turn position into microfluctuation
                    F, batchnr, pos = batch.F, batch.batch, batch.pos
                    affine_pos = torch.matmul(F[batchnr], pos.reshape(-1, 2, 1)).reshape(-1, 2)
                    reals['y'].extend((affine_pos - batch.y).detach().cpu().numpy())
                    preds['y'].extend((affine_pos - preds_temp[0]).detach().cpu().numpy())

                else:
                    reals[var].extend(getattr(batch, var).detach().cpu().numpy())
                    preds[var].extend(preds_temp[j].detach().cpu().numpy()*scaling_factors[j])

            if (i%20) == 0:
                print(f'{i+1}/{n_graphs//bs2}', end='\t')

            for asdf in preds_temp:
                if np.isnan(asdf.detach().cpu().numpy()).any():
                    raise ValueError('nan')

        reals['W'] = np.array(reals['W']).reshape(-1, 1)
        for var in ['y', 'W', 'P', 'D']:
            if var == 'y':
                reals[var] = np.concatenate(reals[var]) #.flatten()
                preds[var] = np.concatenate(preds[var]) #.flatten()
            else:
                reals[var] = np.array(reals[var]) #.flatten()
                preds[var] = np.array(preds[var]) #.flatten()
            print(f'{name} {key} reals[{var}].shape:', reals[var].shape)
            print(f'{name} {key} preds[{var}].shape:', preds[var].shape)

        results_MSE[name][key] = {}
        results_R2[name][key] = {}
        results_frob[name][key] = {}
        print('')
        for var in ['y', 'W', 'P', 'D']:
            err = preds[var] - reals[var]
            print('err.shape', err.shape)
            # sum over all axes except the first one
            err_frob = np.sqrt(np.sum(err**2, axis=tuple(np.arange(1, err.ndim))))
            print('err_frob.shape', err_frob.shape)
            real_frob = np.sqrt(np.sum(preds[var]**2, axis=tuple(np.arange(1, err.ndim))))
            print('real_frob.shape', real_frob.shape)
            results_frob[name][key][var] = np.mean(err_frob)/np.mean(real_frob)
            print(f'results_frob[{name}][{key}][{var}]', results_frob[name][key][var])

            reals[var] = reals[var].flatten()
            preds[var] = preds[var].flatten()
            print(f'{name} {key} reals[{var}].shape (after flatten):', reals[var].shape)
            print(f'{name} {key} preds[{var}].shape (after flatten):', preds[var].shape)

            results_MSE[name][key][var] = mean_squared_error(reals[var], preds[var])
            results_R2[name][key][var] = r2_score(reals[var], preds[var])

print(results_MSE)
print(results_R2)
print(results_frob)

with open('results/final_results/results_MSE_4.pkl', 'wb') as f:
    pickle.dump(results_MSE, f)
with open('results/final_results/results_R2_4.pkl', 'wb') as f:
    pickle.dump(results_R2, f)
with open('results/final_results/results_frob.pkl', 'wb') as f:
    pickle.dump(results_frob, f)

# %% Test on F = I
F_I = torch.tensor([[[1.0, 0], [0, 1.0]]])

#test models
results_I = {}
client = mlflow.tracking.MlflowClient()

run_ids = ['9453aa73b9ee4b42ac9560ff37693d6f',
           'f25933db5e5545388265e2e9261edca3',
           '64e45fceb1eb46b6a977851c7123bca8',
           '0f54a8a568094a3085cc387fe21c3d38',
           'bdca87c393db4da3b387a805426d25d2',
           'a49f45e0cc3743ab914283c4ea0e6d60',
           'f09812771fec478eb1195216f3ae018d',
           ]
model_names = ['GNN',
         'GNN, DA ×1',
         'GNN, DA ×2',
         'EGNN',
         'EGNN, DA ×1',
         'EGNN, DA ×2',
         'SimEGNN',
        #  'EGNNmod2_bigger'
        ]
for run_id, name in zip(run_ids, model_names):
    print('================================================')
    print('testing model', name)
    torch.cuda.empty_cache()
    gc.collect()

    results_I[name] = {}

    # import model

    art_path = client.get_run(run_id).info.artifact_uri[8:]
    print(art_path)

    # Fetch the logged artifacts
    artifacts = client.list_artifacts(run_id)

    # find the files with the weights of the model and the parameters needed to initialize it
    for artifact in artifacts:
        if 'weights.pt' in artifact.path:
            model_path = os.path.join(art_path, artifact.path)
            print(model_path)
        elif artifact.path.endswith('model_init_params.pkl'):
            init_params_path = os.path.join(art_path, artifact.path)
            print(init_params_path)

    # import initialization parameters
    with open(init_params_path, 'rb') as f:
        init_params = pickle.load(f)

    # make sure model definition is imported from the right directory
    sys.path.insert(0, art_path)

    # import model definition
    files = os.listdir(art_path)
    fleur_GNN_definition = [file for file in files if file.startswith('fleur_GNN')]
    if len(fleur_GNN_definition) > 1:
        raise Exception('Multiple fleur_GNN definitions found')
    elif len(fleur_GNN_definition) == 0:
        raise Exception('No fleur_GNN definition found')

    fG = importlib.import_module(fleur_GNN_definition[0].split('.')[0])
    fG = importlib.reload(fG)

    # create model, load params
    model = fG.MyGNN(**init_params)
    model.load_state_dict(torch.load(model_path))
    print(model)

    scaling_factors = eval(mlflow.get_run(run_id).data.params['scaling_factors'])
    print('scaling_factors:', scaling_factors)
    device = torch.device('cuda')
    model.to(device)

    for key in data:
        print(key)

        graph = data[key][0]
        graph.F = F_I
        graph.mean_pos = torch.mean(graph.pos, axis=0, keepdim=True)
        graph.batch = torch.zeros(len(graph.pos), dtype=torch.long)
        graph.to(device)


        with torch.no_grad():
            preds_temp = model(graph)

        MSE = torch.mean((preds_temp[0] - graph.pos)**2)

        # because F=I, predicted position should be equal to initial
        results_I[name][key] = MSE

print(results_I)

with open('results/final_results/results_F=I.pkl', 'wb') as f:
    pickle.dump(results_I, f)

