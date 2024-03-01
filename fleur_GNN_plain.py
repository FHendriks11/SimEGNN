import torch
import torch_geometric as tg
import numpy as np
from torch import Tensor
from torch_geometric.typing import SparseTensor
from typing import Tuple

class EdgeNodeUpdate(tg.nn.MessagePassing):
    """message passing layer that updates: node embedding, edge embedding and returns the messages.
    The message each edge sends is based on the distance between and the embedding of the nodes it connects and the edge embedding.
    The node embedding is updated based on the previous node embedding and the aggregated message received from neighboring nodes.
    The edge embedding is updated based on the message it sends.

    parameters
    ----------
    messagepassing : [type]
        [description]
    """    

    def __init__(self, node_in, edge_in, message_size, node_out, edge_out):
        
        """initialize layer

        parameters
        ----------
        node_in : int
            previous node embedding size
        edge_in: int
            previous edge embedding size
        message_size : int
            size of the message
        node_out : int
            node embedding size after updating
        edge_out: int
            edge embedding size after updating
        """

        super().__init__(aggr='mean')
        self.add_module('mlp_message', torch.nn.Sequential(
                        torch.nn.Linear(6 + 2*node_in + edge_in, message_size),
                        torch.nn.Softplus(),
                        # torch.nn.Linear(message_size, message_size),
                        # torch.nn.Softplus(),
                        ))
        if node_out == 0:
            self.mlp_update = None
        else:
            self.mlp_update = torch.nn.Sequential(
                            torch.nn.Linear(node_in + message_size, node_out),
                            # torch.nn.Softplus(),
                            # torch.nn.Linear(node_out, node_out),
                            )
        if edge_out == 0:
            self.mlp_edge = None
        else:
            self.mlp_edge = torch.nn.Sequential(
                            torch.nn.Linear(message_size, edge_out),
                            # torch.nn.Softplus(),
                            # torch.nn.Linear(edge_out, edge_out),
                            )

        self._message_forward_hooks['hook1'] = self.hook_temp

    def hook_temp(self, net, msg_kwargs, out):
        net.messages = out

    def forward(self, x, edge_index, edge_attr, r, r_init, d_init):
        """[summary]

        parameters
        ----------
        x : torch.tensor, shape [n, node_in]
            current node embedding for each of the n nodes
        edge_index : torch.tensor, shape [2, e]
            indices of all edges
        edge_attr : torch.tensor, shape [e, edge_in]
            edge_attributes of each of the e edges
        r : torch.tensor, shape [n, dim]
            edge vectors
        r_init : torch.tensor, shape [n, dim]
            original edge vectors (i.e. in reference configuration)
        d_init : torch.tensor, shape [n, 1]
            original distances between nodes (i.e. in reference configuration)
        """
        # compute distance for each edge
        d = torch.norm(r, dim=-1, keepdim=True)
        physical_quantity = torch.cat((r, d, r_init, d_init), dim=1)

        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, physical_quantity=physical_quantity)

        if not self.mlp_edge is None:
            edge_attr = self.edge_updater(edge_index, messages=self.messages)
        else:
            edge_attr = torch.tensor([1.0], dtype=torch.float32, device=x.device)


        return x, edge_attr, self.messages

    def message(self, x_i, x_j, physical_quantity, edge_attr):
        """computes message that each edge sends

        parameters
        ----------
        x_i : torch.tensor, shape [e, node_in]
            node embeddings of target nodes
        x_j : torch.tensor, shape [e, node_in]
            node embeddings of source nodes
        physical_quantity : torch.tensor, shape [e, 1]
            some physical quantity associated with the edge between node j and node i (can be distance or strain)
        edge_attr : torch.tensor, shape [e, edge_in]
            edge embeddings
        """
        temp = torch.cat((x_j, x_i, physical_quantity, edge_attr), dim=1)
        return self.mlp_message(temp)

    def update(self, aggr_out, x):
        if self.mlp_update is None:
            return torch.tensor([1.0], dtype=torch.float32, device=x.device)
        else:
            temp = torch.cat((x, aggr_out), dim=1)
            return self.mlp_update(temp)

    def edge_update(self, messages):
        """_summary_

        Parameters
        ----------
        messages : None or torch.tensor, shape [e, message_size]
            message that each edge sends

        Returns
        -------
        torch.tensor
            updated embedding of the edges
        """        
        return self.mlp_edge(messages)
        

class PosUpdate(torch.nn.Module):
    """Layer that updates: the positions of the nodes and the vectors r_ij pointing along each edge.
    The shift in node position is calculated from the node embedding.
    """    

    def __init__(self, node_in):
        
        """initialize layer

        parameters
        ----------
        node_in : int
            size of the node embeddings
        """
        super().__init__()

        self.mlp_shift = torch.nn.Sequential(
                        torch.nn.Softplus(),
                        torch.nn.Linear(node_in, 2),
                        )

    def forward(self, edge_index, pos, r, node_embedding):
        """[summary]

        parameters
        ----------
        edge_index : torch.tensor, shape [2, e]
            indices of all edges
        pos : torch.tensor, shape [n, nr of space dimensions]
            current position of each node
        r : torch.tensor, shape [n, dim]
            edge vectors
        node_embedding: torch.tensor, shape [e, message_size]
            messages sent by each edge
        """  
        shift = self.mlp_shift(node_embedding)
        pos += shift
        r = r + shift[edge_index[1]] - shift[edge_index[0]]
        return pos, r

class MyGNN(torch.nn.Module):
    def __init__(self, layers, noise_std=None, reuse_layers=None):
        """Initialize an E(n) equivariant GNN for a periodic mesh.

        Parameters
        ----------
        layers : tuple of tuples
            each element of the tuple is itself a tuple with 5 elements, which are: node_in, edge_in, message_size, node_out, edge_out
        noise_std : float, optional
            standard deviation of Gaussian noise to add during first position update, by default None
        reuse_layers : tuple of ints, optional
            how often each layer should be applied. If None, each layer will be applied once. By default None
        """        
        
        super().__init__()

        if reuse_layers is not None:
            if len(layers) != len(reuse_layers):
                raise ValueError(f'length of layers (currently {len(layers)}) must be equal to length of reuse_layers (currently {len(reuse_layers)}')
        
        self.num_layers = len(layers)

        self.edgeNodeUpdates = torch.nn.ModuleList(
                [EdgeNodeUpdate(*layer) for layer in layers]
            )
        self.posUpdates = torch.nn.ModuleList(
                [PosUpdate(layer[3]) for layer in layers]
            )

        self.noise_std = noise_std

        if reuse_layers is None:
            self.reuse_layers = (1,) * len(layers)
        else:
            self.reuse_layers = reuse_layers

        self.num_mpsteps = sum(self.reuse_layers)

        # # define batch normalization modules for both the node embeddings and the edge embeddings
        # self.batchnormlayers = torch.nn.ModuleList(
        #     [torch.nn.BatchNorm1d(4) for i in range(79)] +
        #     [torch.nn.BatchNorm1d(4) for i in range(79)]
        # )

        self.energy_mlp = torch.nn.Sequential(
            # torch.nn.Linear(layers[-1][3] + layers[-1][4], 64),
            torch.nn.Linear(layers[-1][2], 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 1),
            )
        message_size = layers[-1][2]  # final message size
        self.mlp_P = torch.nn.Sequential(
                        torch.nn.Linear(message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, 4),
                        )
        
        self.mlp_D = torch.nn.Sequential(
                        torch.nn.Linear(message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, 16),
                        )

    def forward(self, data, verbose=False):
        F, x, edge_index = data.F, data.x, data.edge_index
        edge_attr, pos, r_init, d = data.edge_attr, data.pos, data.r, data.d
        mean_pos, batch = data.mean_pos, data.batch

        # apply F
        pos = torch.matmul(F[batch], pos.reshape(-1, 2, 1)).reshape(-1, 2)
        r = torch.matmul(F[batch[edge_index[0]]], r_init.reshape(-1, 2, 1)).reshape(-1, 2)

        if verbose:
            print(f'{"i":2} {"j":2} {"std m":9} {"mean m":9} {"std x":9} {"mean x":9} {"std e":9} {"mean e":9} {"std pos":9} {"mean pos":9}')
            print(f'{"init":5} {"":9} {"":9} {torch.std(x):9.2e} {torch.mean(x):9.2e} {torch.std(edge_attr):9.2e} {torch.mean(edge_attr):9.2e} {torch.std(pos):9.2e} {torch.mean(pos):9.2e}')

        for i in range(self.num_layers):
            for j in range(self.reuse_layers[i]):

                x, edge_attr, messages = self.edgeNodeUpdates[i](x, edge_index, edge_attr=edge_attr, r=r, r_init=r_init, d_init=d)

                if i == 0 and j == 0:
                    pos, r = self.posUpdates[i](edge_index, pos, r, x)
                else:
                    pos, r = self.posUpdates[i](edge_index, pos, r, x)

                # print('i, j:', i, j)
                
                # activation function
                x = torch.nn.functional.softplus(x) - np.log(2)
                edge_attr = torch.nn.functional.softplus(edge_attr) - np.log(2)

                if verbose: 
                    print(f'{i:2} {j:2} {torch.std(messages):9.2e} {torch.mean(messages):9.2e} {torch.std(x):9.2e} {torch.mean(x):9.2e} {torch.std(edge_attr):9.2e} {torch.mean(edge_attr):9.2e} {torch.std(pos):9.2e} {torch.mean(pos):9.2e}')

                # torch.cuda.synchronize()

        # shift fixed node back in place
        mean_pos_pred = tg.nn.global_mean_pool(pos, batch) # mean predicted position of all nodes per graph, shape: [nr of graphs, 2]
        shift = mean_pos_pred - mean_pos  # how far this mean pos has shifted from where it should be, shape: [nr of graphs, 2]
        pos -= shift[[batch]]  # shift all nodes, such that mean pos is where it should be. shape pos: [nr of graphs*nr of nodes, 2]. shape shift[[batch]]: [nr of graphs*nr of nodes, 2]

        # # pool node embedding per graph
        # x = tg.nn.global_mean_pool(x, batch)

        # get batch nr of each edge
        edge_batch = batch[edge_index[0]]

        # torch.cuda.synchronize()

        # # pool edge embedding per graph
        # edge_attr = tg.nn.global_mean_pool(edge_attr, edge_batch)

        # turn graph level info into an energy
        # energy = self.energy_mlp(torch.cat((x, edge_attr), dim=1))    
        
        # use messages
        messages_pooled = tg.nn.global_mean_pool(messages, edge_batch)   

        energy = self.energy_mlp(messages_pooled)  
        stress = self.mlp_P(messages_pooled).reshape(-1, 2, 2)
        stiffness = self.mlp_D(messages_pooled).reshape(-1,2,2,2,2)
        
        # torch.cuda.synchronize()      
        
        if verbose:
            print(f'{"pool":5} {"":9} {"":9} {torch.std(x):9.2e} {torch.mean(x):9.2e} {torch.std(edge_attr):9.2e} {torch.mean(edge_attr):9.2e} {torch.std(pos):9.2e} {torch.mean(pos):9.2e}')
            print(f'energy: {torch.std(energy):9.2e} {torch.mean(energy):9.2e}')

        return pos, energy, stress, stiffness