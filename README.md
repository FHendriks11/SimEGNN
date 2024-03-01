This repository contains Python code intended to investigate the use of graph neural networks (GNNs) to approximate results of finite element (FE) simulations of a material with square stacked holes. The FE results that are used as a ground truth are obtained using code from Ondrej Rokos' Hole_sheet_model repository. To implement the GNNs, PyTorch Geometric is used.
environment.yml specifies the conda environment that was used, so everything should (absolutely no guarantees) work correctly in an environment with the same package versions.
This repository contains the following Python scripts and Jupyter notebooks:

Creating data set + data exploration:
* data_alternative_deformations.py: creates all possible bifurcation options.
* data_create_graphs.ipynb: turns pandas DataFrame with the Matlab finite element results into PyTorch Geometric Data objects (graphs), includes identifying the holes, removes the bulk nodes and creating wraparound edges.
* data_explore.ipynb: makes various plots of the ground truth training data set.
* data_import_matfiles.py: convert matlab data files into a pandas DataFrame.
* data_plot_graphs.ipynb: make simple plots of the graphs for illustration purposes.

GNN architectures:
* fleur_GNN_plain.py: contains the GNN model with only translation in-/equivariance. Imported in GNN_train.py.
* fleur_GNN.py: contains the GNN model with all the symmetries (optionally also scale in-/equivariance). Imported in GNN_train.py.

Useful functions:
* funcs_helpers.py: some functions that are useful, imported in various other scripts. Including: splitting data in train/validation by trajectory, checking for crossing in edges and replacing integers with other integers.
* funcs_plotting: some functions to plot various quantities.
* funcs_training: functions to train a GNN, called in GNN_train.py.

Training & testing GNNs & visiualizing results:
* GNN_mlflow_results_plots.ipynb: import results from mlflow, combine and make nice plots.
* GNN_results.ipynb: makes deformation plots using trained GNNs, and uses results from GNN_test.py to make spider plots and latex tables.
* GNN_test.py: tests different models and compares them, on various test cases.
* GNN_train.py: trains the GNN.