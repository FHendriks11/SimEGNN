# %%
import pickle
import time
import os
import sys
import importlib
import argparse

import torch
import numpy as np
import torch_geometric as tg
import mlflow
import matplotlib.pyplot as plt

import funcs_training as ft
import funcs_plotting as fp
import funcs_helpers as fh

# %%
parser = argparse.ArgumentParser()
# location data file
parser.add_argument("-d", "--data", type=str)
# location of mlruns directory
parser.add_argument("-m", "--mlflowdir", type=str)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

if args.mlflowdir is  not None:
    abs_path = os.path.abspath(args.mlflowdir)
    mlflow.set_tracking_uri('file://' + os.path.join(abs_path, 'mlruns'))

print('imported libraries')

# Set mlflow experiment
mlflow.set_experiment('GNN_pos_W_P_D_noBifurc')
print('set mlflow experiment to GNN_pos_W_P_D_noBifurc')

mlflow.start_run(run_name='SimEGNN', description='SimEGNN')

# %% Architecture Settings/hyperparameters
# load previous, resume training it (set to None to start with new)
continue_from = None  # run_id of previous run

if continue_from is None:

    plain_model = False  # True  # use model with only translation in/equivariance

    if plain_model:
        layers =  ([(0, 1, 64, 32, 32)]
                   + [(32, 32, 64, 32, 32) for i in range(3)]
                   + [(32, 32, 64, 32, 0)])  # needs final node embedding for final shift
    else:
        layers =  [(0, 1, 64, 32, 32)] + [(32, 32, 64, 32, 32) for i in range(3)] + [(32, 32, 64, 0, 0)]  # in case of empty node attr, no final embeddings
        # layers =  [(1, 0, 64, 32, 32)] + [(32, 32, 64, 32, 32) for i in range(3)] + [(32, 32, 64, 0, 0)] # in case of empty edge attr, no final embeddings
        use_strain = True  #  makes deformation scale equivariant
        scale_r =  True  # makes P/D scale invariant
        mlflow.log_param('use_strain', use_strain)
        mlflow.log_param('scale_r', scale_r)

    print('layers:', layers)
    reuse_layers = (1, 3, 3, 3, 1)

    if plain_model:
        model_init_params = {'layers': layers, 'reuse_layers': reuse_layers}
    else:
        model_init_params = {'layers': layers, 'reuse_layers': reuse_layers, 'use_strain': use_strain, 'scale_r': scale_r}

else:
    run = mlflow.get_run(run_id=continue_from)
    reuse_layers = run.data.params['reuse_layers']
    layers = run.data.params['layers']
    scaling_factor_W = run.data.params['scaling_factor_W']
    try:
        use_strain = run.data.params['use_strain']
    except Exception:
        use_strain = True
mlflow.log_param('layers', layers)
mlflow.log_param('reuse_layers', reuse_layers)
mlflow.log_param('continue_from', continue_from)
mlflow.log_param('plain_model', plain_model)

print('logged settings/hyperparameters in mlflow')

# %% Training settings/hyperparameters
# tuples of optimizer, learning rate, batch size, nr of epochs
bs = 12
lr_schedule = [
               ('Adam', 2.5e-4, bs, bs*10),
               ('Adam', 1.0e-4, bs, bs*50),
               ('Adam', 5.0e-5, bs, bs*30),
               ('Adam', 2.5e-5, bs, bs*30),
               ('Adam', 1.0e-5, bs, bs*5),
               ('Adam', 5.0e-6, bs, bs*5),
               ('Adam', 2.5e-6, bs, bs*5),
               ]

# how much to weigh the losses (pos, W, P, D) relative to each other
relative_weight_losses = [1.0, 1.0, 1.0, 1.0]

grad_clipping = 0.5

mlflow.log_param('lr_schedule', lr_schedule)
mlflow.log_param('relative_weight_losses', relative_weight_losses)
mlflow.log_param('grad_clipping', grad_clipping)

if args.data is not None:
    data_path = args.data
else:
    data_path = 'data\coarseMesh_noBifurcation_5\graphs_coarseMesh_noBifurcation_diameter_0.9_5_noBulkNodes_4.pkl'
mlflow.log_param('dataset', data_path)

# ===================== ALL HYPERPARAMETERS ABOVE =====================

# %% Choose device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda available')
    mlflow.log_param('device', torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print('cuda not available')
    mlflow.log_param('device', 'cpu')

# %% Create model and save architecture details
# remove 'file:///' from path
art_path = mlflow.get_artifact_uri()[8:]
# print('art_path', art_path)
# print('os.path.splitdrive(art_path)', os.path.splitdrive(art_path))
# if there is no drive letter, prepend another slash (e.g. for linux)
if os.path.splitdrive(art_path)[0] == '':
    art_path = '/' + art_path
print('art_path:', art_path)

if continue_from is None:
    if plain_model:
        import fleur_GNN_plain as fg
    else:
        import fleur_GNN as fg

    # create network instance
    model = fg.MyGNN(**model_init_params).to(device=device)

    # save and log parameters used to initialize the model
    path = os.path.join(art_path, 'model_init_params.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model_init_params, f)
    print('Saved initialization parameters of the model to', path)

    # # Save model definition
    module_path = os.path.abspath(fg.__file__)
    mlflow.log_artifact(module_path)
    print('Logged model definition')

else:
    client = mlflow.tracking.MlflowClient()
    # remove 'file:///' from path
    art_path_import = client.get_run(continue_from).info.artifact_uri[8:]
    # if there is no drive letter, prepend another slash
    if os.path.splitdrive(art_path_import)[0] == '':
        art_path_import = '/' + art_path_import

    art_path_import = client.get_run(continue_from).info.artifact_uri[8:]
    # Fetch the logged artifacts
    artifacts = client.list_artifacts(continue_from)

    # find the files with the weights of the model and the parameters needed to initialize it
    for artifact in artifacts:
        if 'weights.pt' in artifact.path:
            model_path = os.path.join(art_path_import, artifact.path)
            print(model_path)
        elif artifact.path.endswith('model_init_params.pkl'):
            init_params_path = os.path.join(art_path_import, artifact.path)
            print(init_params_path)

    with open(init_params_path, 'rb') as f:
        init_params = pickle.load(f)
    mlflow.log_artifact(init_params_path)

    # import model definition
    files = os.listdir(art_path_import)
    fleur_GNN_definition = [file for file in files if file.startswith('fleur_GNN')]
    if len(fleur_GNN_definition) > 1:
        raise ImportError('multiple model definitions found')
    elif len(fleur_GNN_definition < 1):
        raise ImportError('no model definition found to import')
    else:
        fG = importlib.import_module(fleur_GNN_definition[0].split('.')[0])
    model = fG.MyGNN(**init_params)
    module_path = os.path.abspath(fg.__file__)
    mlflow.log_artifact(module_path)

    model.load_state_dict(torch.load(model_path))
    model.to(device)

# Save this script itself
file_path = os.path.abspath(__file__)
print(file_path)
mlflow.log_artifact(file_path)
print('Logged script')

# %%
for name, param in model.named_parameters():
    print(f'{name:35} {str(param.shape):15} {param.numel()}')
n_params = sum(p.numel() for p in model.parameters())
print('Total nr of parameters:', n_params)
mlflow.log_param('nr_parameters', n_params)

print(model)

# %% Prepare training data

# Open data
with open(data_path, 'rb') as f:
    data_list0 = pickle.load(f)
print('First graph:', data_list0[0])
print('Nr of graphs:', len(data_list0))

# %%
# divide into train and validation
trajs = [graph.traj.numpy() for graph in data_list0]
train_inds, val_inds = fh.split_trajs(trajs, 2, split_sizes=(0.7, 0.3))
train_temp = [data_list0[ind] for ind in train_inds]
for i, graph in enumerate(train_temp):
    graph.ind = [i]
val_temp = [data_list0[ind] for ind in val_inds]

val_loader = tg.loader.DataLoader(val_temp, batch_size=16)

# count things
n_nodes = data_list0[0].num_nodes
print('nr of nodes per graph')
print(n_nodes)

n_graphs_train = len(train_temp)
n_graphs_val = len(val_temp)
print('\nnr of graphs, train & val')
print(n_graphs_train)
print(n_graphs_val)

train_nr_of_nodes = n_nodes*n_graphs_train
val_nr_of_nodes = n_nodes*n_graphs_val
print('\ntotal nr of nodes in train & val set')
print(train_nr_of_nodes)
print(val_nr_of_nodes)

loss_fn = torch.nn.MSELoss()

# calculate scaling factors
scaling_factors = [1.0, 1.0, 1.0, 1.0]
W_train = np.array([graph.W.item() for graph in train_temp])
scaling_factors[1] = np.sqrt(np.mean(W_train**2))
scaling_factors[2] = np.sqrt(np.mean([graph.P.numpy()**2 for graph in train_temp]))
scaling_factors[3] = np.sqrt(np.mean([graph.D.numpy()**2 for graph in train_temp]))
mlflow.log_param('scaling_factors', scaling_factors)
scaling_factors = torch.tensor(scaling_factors, device=device)

weight_losses = np.ones(4)
temp = []
for graph in train_temp:
    if hasattr(graph, 'scale_factor'):
        affine_pos = np.matmul(graph.F.numpy(), graph.pos.numpy().T)/graph.scale_factor[0].item()
        real_pos = graph.y.numpy().T/graph.scale_factor[0].item()
    else:
        affine_pos = np.matmul(graph.F.numpy(), graph.pos.numpy().T)
        real_pos = graph.y.numpy().T
    U = affine_pos - real_pos
    temp.append(np.mean(U**2))
weight_losses[0] = np.mean(temp)
print(weight_losses)
weight_losses = weight_losses[0]/weight_losses*relative_weight_losses
print(weight_losses)
mlflow.log_param('weight_losses', weight_losses)
weight_losses = torch.tensor(weight_losses, device=device)

# calculate validation loss before starting training
avg_loss, avg_MSE = ft.val_loop_stress_stiffness(val_loader, model, device,weight_losses, scaling_factors=scaling_factors)

print(f"Avg val losses:", end=" ")
for loss_temp, name in zip(avg_loss, ['pos', 'W', 'P', 'D']):
    mlflow.log_metric('val MSE '+ name + '_scaled', loss_temp, step=-1)
    print(f"{loss_temp:>13.7}", end=" ")
print('')
print(f"Avg val MSE:   ", end=" ")
for loss_temp, name in zip(avg_MSE, ['pos', 'W', 'P', 'D']):
    mlflow.log_metric('val MSE '+ name, loss_temp, step=-1)
    print(f"{loss_temp:>13.7}", end=" ")
print('')

# %% Train
time_temp = time.time()

# train
epoch = 0
ep_cumul = 0
try:
    step = 0
    for optim, lr, batch_size, ep in lr_schedule:
        print(f'{ep} epochs with learning rate {lr}')
        if optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optim == 'AdamW+amsgrad':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
        else:
            raise NotImplementedError(f'Optimizer type {optim} not implemented')

        print('len(train_temp)', len(train_temp))
        train_loader = tg.loader.DataLoader(train_temp, batch_size=batch_size, shuffle=True)
        n_batches_train = len(train_loader)
        n_batches_val = len(val_loader)
        print('\nnr of batches, train & val')
        print(n_batches_train)
        print(n_batches_val)

        for epoch in range(ep_cumul, ep_cumul + ep):
            print('epoch', epoch)  #, end=' ')
            t_start = time.time()

            avg_loss, avg_MSE = ft.train_loop_stress_stiffness(train_loader, model, device, optimizer, weight_losses, scaling_factors=scaling_factors)

            print(f"Avg train losses:", end=" ")
            for loss_temp, name in zip(avg_loss, ['pos', 'W', 'P', 'D']):
                mlflow.log_metric('train MSE '+ name + '_scaled', loss_temp, step=epoch)
                print(f"{loss_temp:>13.7}", end=" ")
            print('')
            print(f"Avg train MSE:   ", end=" ")
            for loss_temp, name in zip(avg_MSE, ['pos', 'W', 'P', 'D']):
                mlflow.log_metric('train MSE '+ name, loss_temp, step=epoch)
                print(f"{loss_temp:>13.7}", end=" ")
            print('')

            print(f'training epoch {epoch} finished in {time.time()-t_start} seconds')

            avg_loss, avg_MSE = ft.val_loop_stress_stiffness(val_loader, model, device, weight_losses, scaling_factors=scaling_factors)

            print(f"Avg val losses:", end=" ")
            for loss_temp, name in zip(avg_loss, ['pos', 'W', 'P', 'D']):
                mlflow.log_metric('val MSE '+ name + '_scaled', loss_temp, step=epoch)
                print(f"{loss_temp:>13.7}", end=" ")
            print('')
            print(f"Avg val MSE:   ", end=" ")
            for loss_temp, name in zip(avg_MSE, ['pos', 'W', 'P', 'D']):
                mlflow.log_metric('val MSE '+ name, loss_temp, step=epoch)
                print(f"{loss_temp:>13.7}", end=" ")
            print('')

        ep_cumul += ep
    mlflow.log_param('nr_epochs', epoch)

except BaseException as e:
    # raise(e)
    print('Caught exception, stopping training')
    print(repr(e))
    mlflow.log_param('exception', repr(e))
    mlflow.log_param('nr_epochs', epoch)

print('total time:', time.time() - time_temp)

# %% Save things
# save parameters of model
path = os.path.join(art_path, 'model_weights.pt')
torch.save(model.state_dict(), path)
print('Saved model parameters to', path)

# # Save entire model
path = os.path.join(art_path, 'model.pt')
torch.save(model, path)
print('Saved model to', path)

# %% Calculate W, P, D  on validation data
# Predict stuff
W_pred_arr = []
P_pred_arr = []
D_pred_arr = []
W = []
P = []
D = []

val_loader = tg.loader.DataLoader(val_temp, batch_size=8)

# Predict stuff for validation data
for batch in val_loader:
    batch.to(device=device)

    batch.F.requires_grad = True

    # calculate energy density (total energy)
    _, W_pred, P_pred, D_pred = model(batch)

    batch_size = len(W_pred)

    # calculate total energy (energy density?)
    W_pred = W_pred*scaling_factors[1]
    P_pred = P_pred*scaling_factors[2]
    D_pred = D_pred*scaling_factors[3]

    # append predictions
    W_pred_arr.extend(W_pred.detach().cpu().numpy().tolist())
    P_pred_arr.extend(P_pred.detach().cpu().numpy().tolist())
    D_pred_arr.extend(D_pred.detach().cpu().numpy().tolist())

    # append real values
    W.extend(batch.W.detach().cpu().numpy().tolist())
    P.extend(batch.P.detach().cpu().numpy().tolist())
    D.extend(batch.D.detach().cpu().numpy().tolist())

W_pred_arr = np.array(W_pred_arr)[:, 0]
P_pred_arr = np.array(P_pred_arr)
D_pred_arr = np.array(D_pred_arr)
W = np.array(W)
P = np.array(P)
D = np.array(D)

# make crossplots
fig = fp.plot_W(W_pred_arr, W)
mlflow.log_figure(fig, 'crossplot_energy_density.png')
mlflow.log_figure(fig, 'crossplot_energy_density.svg')
print('plotted W')
plt.close()

fig = fp.plot_P(P_pred_arr, P)
mlflow.log_figure(fig, 'crossplot_P_components.png')
mlflow.log_figure(fig, 'crossplot_P_components.svg')
print('plotted P')
plt.close()

fig = fp.plot_D(D_pred_arr, D)
mlflow.log_figure(fig, 'crossplot_D_components.png')
mlflow.log_figure(fig, 'crossplot_D_components.svg')
print('plotted D')
plt.close()

# %% Log metrics
temp = np.mean((W_pred_arr - W)**2)
mlflow.log_metric('val MSE W', temp)
print('logged validation MSE W to mlflow')

temp = np.mean((P_pred_arr - P)**2)
mlflow.log_metric('val MSE P', temp)

temp = np.mean((D_pred_arr - D)**2)
mlflow.log_metric('val MSE D', temp)

print('logged validation MSE P, D to mlflow')
