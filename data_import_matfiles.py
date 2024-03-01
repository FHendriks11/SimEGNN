# %% import libraries
import pickle
import os

import pandas as pd
import numpy as np
from scipy.io import loadmat

# %% paths
diameter = 0.85

# path to directory with the results to import
results_dir = ...

# path to put DataFrame
df_path = f'data\coarseMesh_noBifurcation_5\dataframe_coarseMesh_noBifurcation_diameter_{diameter}'

constant_mesh = True #  False  #

# %%
# find files
list_of_files = os.listdir(results_dir)
print(len(list_of_files), 'files found')

# get list of results and mesh info files
results_files = []
mesh_files = []
for file in list_of_files:
    if file.endswith('.mat') and file.startswith('results'):
        results_files.append(file)
    elif file.endswith('.mat') and file.startswith('mesh_info'):
        mesh_files.append(file)

print(len(results_files), 'results files found')
print(len(mesh_files), 'mesh files found')

if not constant_mesh:
    # make sure both lists of files are sorted the same way
    G_results = np.array([file.split('_')[-1] for file in results_files])
    G_meshes = np.array([file.split('_')[-1] for file in mesh_files])

    assert (G_results == G_meshes).all()

# %%
# read all results
data = {'trajectory': []}
i=0
for file in results_files:
    mat = loadmat(os.path.join(results_dir, file))
    # append all results in this file to the rest of the results
    for key in mat:
        if isinstance(mat[key], np.ndarray):
            if key in data:
                data[key].append(mat[key])
            else:
                data[key] = [mat[key]]
    data['trajectory'].append(np.full(mat['Time'].shape[-1], i))
    i += 1

# read all meshes
if not constant_mesh:
    pos = []
    for file in mesh_files:
        mat = loadmat(os.path.join(results_dir, file))
        # append all results in this file to the rest of the results
        pos.append(mat['p_new'])
    pos = np.stack(pos)
    pos.shape

# check all U have the same length
temp = np.array([elem.shape[0] for elem in data['U']])
vals, counts = np.unique(temp, return_counts=True)
print('unique lengths of U (should be only one value):', vals)
print('counts:', counts)
assert len(vals) == 1

# check if the imported data makes sense
for key in data:
    print(f'{key:12} {repr(len(data[key])):6} {repr(data[key][0].shape):17} {repr(type(data[key])):25} {repr(type(data[key][0])):25}')

# make list of length of each trajectory
traj_lengths = [traj.shape[-1] for traj in data['Time']]

# duplicate pos for each step in a trajectory
if not constant_mesh:
    print('pos.shape:', pos.shape)
    pos = np.repeat(pos, repeats=traj_lengths, axis=0)
    print('pos.shape:', pos.shape, '(after duplicating pos for each step in a trajectory)')

# concatenate all data
for key in data:
    # make all arrays from the same trajectory the same length (of Time)
    # (throw away last step(s) if necessary, these were not completed anyway if the lengths are unequal)
    for i, traj in enumerate(data[key]):
        data[key][i] = data[key][i][...,:traj_lengths[i]]

    # concatenate along last dim, which is the time steps
    data[key] = np.concatenate(data[key], axis=-1)

# make time steps the first dimension
for key in data:
    n = len(data[key].shape)  # nr of dims
    data[key] = np.transpose(data[key], [n-1] + list(range(n-1)))

# flatten Time, W and bifurc arrays
for key in ['Time', 'W', 'bifurc']:
    data[key] = data[key].flatten()

if not constant_mesh:
    # add position too
    data['pos'] = pos

# check if the resulting data makes sense
print('\nshapes in data dict:')
for key in data:
    print(key, len(data[key]), data[key][0].shape)

# Turn data into lists
data2 = {}
for key in data:
    data2[key] = list(data[key])

# turn data into a pandas dataframe
df = pd.DataFrame(data2)
df = df.astype({'bifurc': bool}, copy=False)
print('\ndtypes in DataFrame:')
print(df.dtypes)
df

# %% save dataframe to file
path = df_path + '.pkl'
if os.path.exists(path):
    raise FileExistsError(f'{path} already exists')
else:
    with open(path, 'wb') as f:
        pickle.dump(df, f)