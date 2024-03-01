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

    def __init__(self, node_in, edge_in, message_size, node_out, edge_out, use_strain=True):
        
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
        use_strain: 
            use strain instead of distance to make the network scale invariant/equivariant. By default True.
        """

        super().__init__(aggr='mean')
        if use_strain:
            self.add_module('mlp_message', torch.nn.Sequential(
                            torch.nn.Linear(1 + 2*node_in + edge_in, message_size),
                            torch.nn.Softplus(),
                            # torch.nn.Linear(message_size, message_size),
                            # torch.nn.Softplus(),
                            ))
        else:  # if using no strain, r_ref is also used, so input is larger
            self.add_module('mlp_message', torch.nn.Sequential(
                            torch.nn.Linear(2 + 2*node_in + edge_in, message_size),
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
        self.use_strain = use_strain

    def hook_temp(self, net, msg_kwargs, out):
        net.messages = out

    def forward(self, x, edge_index, edge_attr, r, d_init):
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
        d_init : torch.tensor, shape [n, 1]
            original distances between nodes
        """
        # compute distance for each edge
        d = torch.norm(r, dim=-1, keepdim=True)

        if self.use_strain:
            # compute strain for each edge
            strain = (d - d_init)/d_init
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr, physical_quantity=strain)
        else:
            physical_quantity = torch.cat((d, d_init), dim=1)
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr, physical_quantity=physical_quantity)

        if not self.mlp_edge is None:
            edge_attr = self.edge_updater(edge_index, messages=self.messages)
        else:
            edge_attr = torch.tensor([1.0], dtype=torch.float32, 
                                     device=x.device)

        if self.use_strain:
            return x, edge_attr, self.messages, strain
        else:
            return x, edge_attr, self.messages, None

    def message(self, x_i, x_j, physical_quantity, edge_attr):
        """computes message that each edge sends

        parameters
        ----------
        x_i : torch.tensor, shape [e, node_in]
            node embeddings of target nodes
        x_j : torch.tensor, shape [e, node_in]
            node embeddings of source nodes
        physical_quantity : torch.tensor, shape [e, p]
            some physical quantity associated with the edge between node j and node i (can be distance + reference distance (p=2) or strain (p=1))
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
        

class PosUpdate(tg.nn.MessagePassing):
    """message passing layer that updates: the positions of the nodes and the vectors r_ij pointing along each edge.
    The shift in node position is calculated from a contribution of each incoming edge, this contribution depends on the message of this edge and the edge vector (which points from the location of the source node to the location of the target node).

    parameters
    ----------
    messagepassing : [type]
        [description]
    """    

    def __init__(self, message_size, use_strain=True):
        
        """initialize layer

        parameters
        ----------
        message_size : int
            size of the message
        use_strain: 
            use strain instead of distance to make the network scale invariant/equivariant. By default True.
        """
        super().__init__(aggr='mean')
        self.mlp_shift = torch.nn.Sequential(
                        # torch.nn.Linear(message_size, message_size),
                        # torch.nn.Softplus(),
                        torch.nn.Linear(message_size, 1),
                        )
        self.use_strain = use_strain

    def forward(self, edge_index, pos, r, messages, strain=None, noise_std=None):
        """[summary]

        parameters
        ----------
        edge_index : torch.tensor, shape [2, e]
            indices of all edges
        r : torch.tensor, shape [n, dim]
            edge vectors
        pos : torch.tensor, shape [n, nr of space dimensions]
            current position of each node
        messages: torch.tensor, shape [e, message_size]
            messages sent by each edge
        strain: torch.tensor, shape [e,], optional
            strain along each edge. By default None.
        """  
        if strain is None and self.use_strain:
            raise ValueError('use_strain=True, so forward need strain input')
        elif strain is not None and not self.use_strain:
            raise ValueError('strain input given, but self.use_strain=False')
        
        shift = self.propagate(edge_index, pos=pos, r=r, messages=messages, strain=strain)
        if noise_std is not None:
            shift = shift + torch.normal(0.0, noise_std, size=shift.shape).to(shift.device)
        pos += shift
        r = self.edge_updater(edge_index, shift=shift, r=r)

        return pos, r

    def message(self, r, messages, strain):
        if strain is None:
            return r*torch.nn.Tanh()(self.mlp_shift(messages))
        else:
            return r*torch.nn.Tanh()(strain*self.mlp_shift(messages))

    def update(self, aggr_out, pos):
        return aggr_out

    def edge_update(self, shift_i, shift_j, r):
        return r + shift_i - shift_j

def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                        sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    # mask = idx_i != idx_k  # Remove i == k triplets.
    # idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()  # [mask]
    idx_ji = adj_t_row.storage.row()  # [mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

class Predict2ndOrderTensor(torch.nn.Module):
    """
    Predicts a second order tensor for every node.
    """    

    def __init__(self, message_size):
        
        """initialize layer

        parameters
        ----------
        message_size : int
            size of the message
        """
        super().__init__()

        self.mlp_P = torch.nn.Sequential(
                        torch.nn.Linear(2*message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, 1),
                        )
        print('message_size:', message_size)

    def forward(self, edge_index, messages, num_nodes, r):
        """A layer that outputs a second-order tensor for each node, which transforms equivariantly with a r under a change in the coordinate system.

        Parameters
        ----------
        edge_index : torch.tensor of ints
            _description_
        messages : torch.tensor, shape [nr of edges, message size]
            the messages that resulted from the message-passing
        num_nodes : int
            nr of nodes in this batch
        r : torch tensor
            edge vectors

        Returns
        -------
        tensor
            a second order tensor for each node.
        """      
        # torch.cuda.synchronize()

        # print('edge_index.shape:', edge_index.shape)  
        # print('num_nodes:', num_nodes)  
        _, _, _, idx_j, _, idx_kj, idx_ji = triplets(edge_index, num_nodes)
        # print('idx_kj:', idx_kj)
        # print('idx_j:', idx_j)
        # print('idx_ji:', idx_ji)
        # print('messages.shape', messages.shape)

        # torch.cuda.synchronize()

        temp = torch.cat((messages[idx_kj], messages[idx_ji]), dim=1)
        m_ijk = self.mlp_P(temp)  # a scalar for each triplet
        dyad = torch.einsum('ij,ik->ikj', r[idx_kj], r[idx_ji])
        temp = m_ijk*dyad.reshape(-1, 4)

        # print('m_ijk.shape', m_ijk.shape)
        # print('dyad.shape', dyad.shape)
        # print('temp.shape', temp.shape)

        # torch.cuda.synchronize()

        # print('=======================')
        # print('idx_j:', idx_j)
        # print('=======================')
        # print('max(idx_j)', torch.max(idx_j))
        # print('min(idx_j)', torch.min(idx_j))

        tensor = tg.nn.global_mean_pool(temp, idx_j).reshape(-1, 2, 2)
        # print('idx_j.shape:\t', idx_j.shape)
        # print('idx_kj.shape:\t', idx_kj.shape)
        # print('idx_ji.shape:\t', idx_ji.shape)
        # print('temp.shape:\t', temp.shape)
        # print('m_ijk.shape:\t', m_ijk.shape)
        # print('dyad.shape:\t', dyad.shape)
        # print('temp.shape:\t', temp.shape)
        # print('tensor.shape:\t', tensor.shape)
        
        # torch.cuda.synchronize()
        
        return tensor


class Predict4thOrderTensor(torch.nn.Module):
    """
    Predicts a fourth order tensor for every node.
    """    

    def __init__(self, message_size):
        
        """initialize layer

        parameters
        ----------
        message_size : int
            size of the message
        """
        super().__init__()
        
        self.mlp_D = torch.nn.Sequential(
                        torch.nn.Linear(2*message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, message_size),
                        torch.nn.Softplus(),
                        torch.nn.Linear(message_size, 1),
                        )

    def forward(self, edge_index, messages, num_nodes, tensors):
        """A layer that outputs a fourth-order tensor for each node, which transforms equivariantly with r under a change in the coordinate system.
        Created from a sum of outer products of pairs of 2nd order tensors.

        Parameters
        ----------
        edge_index : torch.tensor of ints
            _description_
        messages : torch.tensor, shape [nr of edges, message size]
            the messages that resulted from the message-passing
        num_nodes : int
            nr of nodes in this batch
        tensors : torch tensor, shape [nr of nodes, 2, 2]
            a second order tensor for each node

        Returns
        -------
        tensor
            a 4th order tensor for each node.
        """        
        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes)
        # print('idx_kj:', idx_kj)
        # print('idx_ji:', idx_ji)

        temp = torch.cat((messages[idx_kj], messages[idx_ji]), dim=1)
        m_ijk = self.mlp_D(temp)  # a scalar for each triplet
        dyad = torch.einsum('ijk,ilm->ijklm', tensors[idx_i], tensors[idx_k])
        temp = m_ijk*dyad.reshape(-1, 16)

        tensor = tg.nn.global_mean_pool(temp, idx_j).reshape(-1, 2, 2, 2, 2)

        # print('idx_j.shape:\t', idx_j.shape)
        # print('idx_kj.shape:\t', idx_kj.shape)
        # print('idx_ji.shape:\t', idx_ji.shape)
        # print('temp.shape:\t', temp.shape)
        # print('m_ijk.shape:\t', m_ijk.shape)
        # print('dyad.shape:\t', dyad.shape)
        # print('temp.shape:\t', temp.shape)
        # print('tensor.shape:\t', tensor.shape)

        return tensor

if __name__ == "__main__":
    # test data
    edge_index = torch.tensor([[0, 1, 2, 3, 3, 1, 2, 3, 1, 0], [1, 2, 3, 1, 0, 0, 1, 2, 3, 3]], 
                              dtype=torch.long)
    messages = torch.tensor([[0.2, 0.2, 1], 
                             [1,2,3],
                             [4,5,6],
                             [0.1, 0.1, 0.1],
                             [0.9, 1.2, 4],
                             [1.2, 0.2, 1], 
                             [1,3.2,3],
                             [4,0.2,6],
                             [1.1, 0.1, 0.1],
                             [1.9, 1.2, 4]
                             ])
    num_nodes = 4
    r = torch.tensor([[0.1, 0.1], 
                      [0.1, -0.1],
                      [4.2, -3.0],
                      [2, 0],
                      [0, 1.5],
                      [-0.1, 0.1],
                      [-0.1, -0.1], 
                      [-4.2, 3.0],
                      [-2, -0],
                      [-0, -1.5]
                      ])

    transform = np.random.default_rng().random(size=(2,2))
    # transform = [[2.0, 0],[0, 1]]
    transform = torch.tensor(transform, dtype=torch.float32)
    
    # test 2nd order layer
    # 1. Apply layer
    P_layer = Predict2ndOrderTensor(3)
    P_pred1 = P_layer(edge_index, messages, num_nodes, r)
    print('P_pred1:', P_pred1)

    # 2. Apply transform to input, then apply layer
    r2 = torch.einsum('ij,kj->ki', transform, r)
    P_pred2 = P_layer(edge_index, messages, num_nodes, r2)
    print('P_pred2:', P_pred2)

    # 3: Apply layer, then apply transform to output
    P_pred3 = torch.einsum('lj,ijk,km->ilm', transform, P_pred1, transform.T)
    print('P_pred3:', P_pred3)

    # should be the same
    print('P equivariant?', torch.isclose(P_pred2, P_pred3).all())

    # test 4th order layer
    # 1. Apply layer
    D_layer = Predict4thOrderTensor(3)
    D_pred1 = D_layer(edge_index, messages, num_nodes, P_pred1)
    print('D_pred1:', D_pred1)

    # 2. Apply transform to input, then apply layer
    D_pred2 = D_layer(edge_index, messages, num_nodes, P_pred2)
    print('D_pred2:', D_pred2)

    # result 2: apply transform to output
    D_pred3 = torch.einsum('nj,ok,pl,qm,ijklm->inopq', transform, transform, transform, transform, D_pred1)
    print('D_pred3:', D_pred3)

    # should be the same
    print('D equivariant?', torch.isclose(D_pred2, D_pred3).all())

    print('max diff', torch.max(torch.abs(D_pred2 - D_pred3)))
    print('avg diff', torch.mean(torch.abs(D_pred2 - D_pred3)))
            

class MyGNN(torch.nn.Module):
    def __init__(self, layers, noise_std=None, reuse_layers=None, use_strain=True, scale_r=False):
        """Initialize an E(n) equivariant GNN for a periodic mesh.

        Parameters
        ----------
        layers : tuple of tuples
            each element of the tuple is itself a tuple with 5 elements, which are: node_in, edge_in, message_size, node_out, edge_out
        noise_std : float, optional
            standard deviation of Gaussian noise to add during first position update, by default None
        reuse_layers : tuple of ints, optional
            how often each layer should be applied. If None, each layer will be applied once. By default None
        use_strain: 
            use strain instead of distance to make the network scale invariant/equivariant. By default True.
        scale_r: 
            scale edge vectors when used for the computation of higher order tensors (P and D). Necessary for scale invariance. By default False.
        """        
        
        super().__init__()

        if reuse_layers is not None:
            if len(layers) != len(reuse_layers):
                raise ValueError(f'length of layers (currently {len(layers)}) must be equal to length of reuse_layers (currently {len(reuse_layers)}')
        
        self.num_layers = len(layers)

        self.use_strain = use_strain
        self.scale_r = scale_r

        self.edgeNodeUpdates = torch.nn.ModuleList(
                [EdgeNodeUpdate(*layer, use_strain=use_strain) for layer in layers]
            )
        self.posUpdates = torch.nn.ModuleList(
                [PosUpdate(layer[2], use_strain=use_strain) for layer in layers]
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

        self.stress_layer = Predict2ndOrderTensor(layers[-1][2])
        self.stiffness_layer1 = Predict2ndOrderTensor(layers[-1][2])
        self.stiffness_layer2 = Predict4thOrderTensor(layers[-1][2])

    def forward(self, data, verbose=False):
        F, x, edge_index = data.F, data.x, data.edge_index
        edge_attr, pos, r, d = data.edge_attr, data.pos, data.r, data.d
        mean_pos, batch = data.mean_pos, data.batch

        # apply F
        pos = torch.matmul(F[batch], pos.reshape(-1, 2, 1)).reshape(-1, 2)
        r = torch.matmul(F[batch[edge_index[0]]], r.reshape(-1, 2, 1)).reshape(-1, 2)

        if verbose:
            print(f'{"i":2} {"j":2} {"std m":9} {"mean m":9} {"std x":9} {"mean x":9} {"std e":9} {"mean e":9} {"std pos":9} {"mean pos":9}')
            print(f'{"init":5} {"":9} {"":9} {torch.std(x):9.2e} {torch.mean(x):9.2e} {torch.std(edge_attr):9.2e} {torch.mean(edge_attr):9.2e} {torch.std(pos):9.2e} {torch.mean(pos):9.2e}')

        for i in range(self.num_layers):
            for j in range(self.reuse_layers[i]):

                x, edge_attr, messages, strain = self.edgeNodeUpdates[i](x, edge_index, edge_attr=edge_attr, r=r, d_init=d)

                if i == 0 and j == 0:
                    pos, r = self.posUpdates[i](edge_index, pos, r, messages, strain, noise_std=self.noise_std)
                else:
                    pos, r = self.posUpdates[i](edge_index, pos, r, messages, strain)

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
        
        # torch.cuda.synchronize()      
        
        if verbose:
            print(f'{"pool":5} {"":9} {"":9} {torch.std(x):9.2e} {torch.mean(x):9.2e} {torch.std(edge_attr):9.2e} {torch.mean(edge_attr):9.2e} {torch.std(pos):9.2e} {torch.mean(pos):9.2e}')
            print(f'energy: {torch.std(energy):9.2e} {torch.mean(energy):9.2e}')

        num_nodes = len(pos)
        # print('num_nodes', num_nodes)
        # print('edge_index', edge_index)
        # print('messages', messages)
        # print('r', r)

        if self.scale_r:
            r = r/torch.norm(r, dim=-1, keepdim=True)
        stress_per_node = self.stress_layer(edge_index, messages, num_nodes, r)

        stress = tg.nn.global_mean_pool(stress_per_node.reshape(-1, 4), batch).reshape(-1, 2, 2)

        stiffness_per_node = self.stiffness_layer1(edge_index, messages, num_nodes, r) # intermediate step
        stiffness_per_node = self.stiffness_layer2(edge_index, messages, num_nodes, stiffness_per_node)

        stiffness = tg.nn.global_mean_pool(stiffness_per_node.reshape(-1, 16), batch).reshape(-1, 2, 2, 2, 2)

        return pos, energy, stress, stiffness