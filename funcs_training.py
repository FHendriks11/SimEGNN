import torch
import torch_geometric as tg
import numpy as np
import mlflow

def train_loop(dataloader, model, loss_fn1, loss_fn2, device, optimizer, weight_losses, second_order=False, scaling_factor_W=1, sobolev=False, verbose=False, grad_clipping=None):
    model.train()
    loss_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)
        batch.F.requires_grad = True
        batch_size = len(batch.F)

        if sobolev:
            # Compute predictions
            pos_pred, W_pred = model(batch)
            W_pred2 = W_pred*scaling_factor_W
            # calculate stress P
            P_pred = torch.autograd.grad(W_pred2, batch.F, grad_outputs=torch.ones_like(W_pred2),create_graph=True, retain_graph=True)[0]

            # # calculate D
            # grad_outputs = torch.zeros(4, batch_size, 2, 2, device=device)
            # grad_outputs[0, :, 0, 0] = 1
            # grad_outputs[1, :, 0, 1] = 1
            # grad_outputs[2, :, 1, 0] = 1
            # grad_outputs[3, :, 1, 1] = 1
            # D_pred = torch.autograd.grad(P_pred, batch.F,
            #     grad_outputs=grad_outputs, is_grads_batched=True, retain_graph=True, create_graph=True)[0]
            # D_pred = torch.transpose(D_pred, 0, 1).reshape(-1, 2, 2, 2, 2)

            # calculate D
            grad_outputs = torch.ones(batch_size).to(device=device)
            D_pred = torch.empty(len(batch.F), 2, 2, 2, 2, device=device)
            for j in range(2):
                for k in range(2):
                    D_pred[:, j, k] = torch.autograd.grad(P_pred[:, j, k], batch.F, grad_outputs=grad_outputs, retain_graph=True,
                        create_graph=True)[0]

            # Compute losses
            losses = torch.empty(4, device=device)
            losses[0] = loss_fn1(pos_pred, batch.y)  #, batch.batch)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)
            losses[2] = loss_fn2(P_pred, batch.P)
            losses[3] = loss_fn2(D_pred, batch.D)
        else:
            pos_pred, W_pred = model(batch)
            losses = torch.empty(2, device=device)
            losses[0] = loss_fn1(pos_pred, batch.y)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)

        # Backpropagation
        loss = torch.sum(losses*weight_losses)
        optimizer.zero_grad()
        if second_order:
            # hessian needs graph for backprop
            loss.backward(create_graph=True)
        else:
            loss.backward()

        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
        optimizer.step()

        for param in model.parameters():
            param.grad = None

        # check for nan values of losses
        losses2 = [elem.item() for elem in losses*weight_losses]
        losses = [elem.item() for elem in losses]
        loss_per_batch.append(losses)
        if np.isnan(losses).any():
            raise ValueError('Loss')

        if verbose:
            grad_sq = sum((p.grad**2).sum().item() for p in model.parameters())
            print(f'norm grad: {np.sqrt(grad_sq)}')

        log_n_steps = 1 if verbose else 20
        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'total {loss:>13.7} [{i+1}/{len(dataloader)}]')

        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses2, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'(scaled loss components)')

    loss_per_batch = np.asarray(loss_per_batch)
    avg_loss = np.mean(loss_per_batch, axis=0)

    return avg_loss

def train_loop_stress(dataloader, model, loss_fn1, loss_fn2, loss_fn3, device, optimizer, weight_losses, scaling_factor_W=1, scaling_factor_P=1, verbose=False, grad_clipping=None):
    model.train()
    loss_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)
        batch.F.requires_grad = True
        batch_size = len(batch.F)

        pos_pred, W_pred, P_pred = model(batch)
        losses = torch.empty(3, device=device)
        losses[0] = loss_fn1(pos_pred, batch.y)
        losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)
        losses[2] = loss_fn3(P_pred, batch.P/scaling_factor_P)

        # Backpropagation
        loss = torch.sum(losses*weight_losses)
        optimizer.zero_grad()

        loss.backward()

        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
        optimizer.step()

        for param in model.parameters():
            param.grad = None

        # check for nan values of losses
        losses2 = [elem.item() for elem in losses*weight_losses]
        losses = [elem.item() for elem in losses]
        loss_per_batch.append(losses)
        if np.isnan(losses).any():
            raise ValueError('Loss')

        if verbose:
            grad_sq = sum((p.grad**2).sum().item() for p in model.parameters())
            print(f'norm grad: {np.sqrt(grad_sq)}')

        log_n_steps = 1 if verbose else 20
        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'total {loss:>13.7} [{i+1}/{len(dataloader)}]')

        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses2, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'(scaled loss components)')

    loss_per_batch = np.asarray(loss_per_batch)
    avg_loss = np.mean(loss_per_batch, axis=0)

    return avg_loss


def train_loop_stress_stiffness(dataloader, model, device, optimizer, weight_losses, loss_funcs=None, scaling_factors=[1.0, 1.0, 1.0, 1.0], verbose=False, grad_clipping=None):
    if loss_funcs is None:
        loss_funcs = [torch.nn.MSELoss()]*4
    model.train()

    weighted_loss_per_batch = []
    for i, batch in enumerate(dataloader):
        batch.to(device=device)
        batch.F.requires_grad = True
        batch_size = len(batch.F)

        pos_pred, W_pred, P_pred, D_pred = model(batch)

        pos_target = batch.y/scaling_factors[0]

        # per graph, scale pos back by scale_factor
        if hasattr(batch, 'scale_factor'):
            pos_pred /= batch.scale_factor[batch.batch].reshape(-1, 1)
            pos_target /= batch.scale_factor[batch.batch].reshape(-1, 1)

        losses = torch.empty(4, device=device)
        losses[0] = loss_funcs[0](pos_pred, pos_target)
        losses[1] = loss_funcs[1](W_pred[:, 0], batch.W/scaling_factors[1])
        losses[2] = loss_funcs[2](P_pred, batch.P/scaling_factors[2])
        losses[3] = loss_funcs[3](D_pred, batch.D/scaling_factors[3])

        # Backpropagation
        weighted_losses = losses*weight_losses
        total_loss = torch.sum(weighted_losses)
        optimizer.zero_grad()

        total_loss.backward()

        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
        optimizer.step()

        for param in model.parameters():
            param.grad = None

        # check for nan values of losses
        MSEs = losses*scaling_factors**2
        MSEs = [elem.item() for elem in MSEs]
        weighted_losses = [elem.item() for elem in weighted_losses]
        weighted_loss_per_batch.append(weighted_losses)
        if np.isnan(weighted_losses).any():
            raise ValueError('Loss')

        if verbose:
            grad_sq = sum((p.grad**2).sum().item() for p in model.parameters())
            print(f'norm grad: {np.sqrt(grad_sq)}')

        log_n_steps = 1 if verbose else 20
        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(weighted_losses, ['pos', 'W', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'(Weighted loss components, total {total_loss:>13.7})')
        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(MSEs, ['pos', 'W', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'[(MSE) {i+1}/{len(dataloader)}]')

    weighted_loss_per_batch = np.asarray(weighted_loss_per_batch)
    avg_loss = np.mean(weighted_loss_per_batch, axis=0)
    avg_MSE = avg_loss/weight_losses.cpu().numpy()*scaling_factors.cpu().numpy()**2

    return avg_loss, avg_MSE

def train_loop_simple(dataloader, model, loss_fn1, loss_fn2, device, optimizer, weight_losses, scaling_factor_W=1):
    model.train()
    loss_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)
        batch.F.requires_grad = True
        batch_size = len(batch.F)

        pos_pred, W_pred = model(batch)
        losses = torch.empty(2, device=device)
        losses[0] = loss_fn1(pos_pred, batch.y)
        losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)

        # Backpropagation
        loss = torch.sum(losses*weight_losses)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # check for nan values of losses
        losses2 = [elem.item() for elem in losses*weight_losses]
        losses = [elem.item() for elem in losses]
        loss_per_batch.append(losses)
        if np.isnan(losses).any():
            raise ValueError('Loss')

        log_n_steps = 20
        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'total {loss:>13.7} [{i+1}/{len(dataloader)}]')

        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses2, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'(scaled loss components)')

    loss_per_batch = np.asarray(loss_per_batch)
    avg_loss = np.mean(loss_per_batch, axis=0)

    return avg_loss

def train_loop_debug(dataloader, model, loss_fn1, loss_fn2, device, optimizer, weight_losses, scaling_factor_W=1, sobolev=False):
    print('WARNING! using the debugging training loop. The model will not be trained.')
    model.train()
    loss_per_batch = []
    gradnorm_per_batch = []
    inds_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)
        batch.F.requires_grad = True
        batch_size = len(batch.F)

        if sobolev:
            # Compute predictions
            pos_pred, W_pred = model(batch)
            W_pred2 = W_pred*scaling_factor_W
            # calculate stress P
            P_pred = torch.autograd.grad(W_pred2, batch.F, grad_outputs=torch.ones_like(W_pred2),create_graph=True, retain_graph=True)[0]

            # calculate D
            grad_outputs = torch.ones(batch_size).to(device=device)
            D_pred = torch.empty(len(batch.F), 2, 2, 2, 2, device=device)
            for j in range(2):
                for k in range(2):
                    D_pred[:, j, k] = torch.autograd.grad(P_pred[:, j, k], batch.F, grad_outputs=grad_outputs, retain_graph=True,
                        create_graph=True)[0]

            # Compute losses
            losses = torch.empty(4, device=device)
            losses[0] = loss_fn1(pos_pred, batch.y)  #, batch.batch)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)
            losses[2] = loss_fn2(P_pred, batch.P)
            losses[3] = loss_fn2(D_pred, batch.D)
        else:
            pos_pred, W_pred = model(batch)
            losses = torch.empty(2, device=device)
            losses[0] = loss_fn1(pos_pred, batch.y)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)

        # Backpropagation through all losses separately
        gradnorms = []
        for loss, name, weight_loss in zip(losses, ['pos', 'W_scaled', 'P', 'D'], weight_losses):
            loss = loss*weight_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            grad_sq = sum((p.grad**2).sum().item() for p in model.parameters())
            gradnorm = np.sqrt(grad_sq)
            print(f'{name} {gradnorm:>13.7},', end=' ')
            # optimizer.step()
            gradnorms.append(gradnorm)
        print(f'norm of gradient')

        for param in model.parameters():
            param.grad = None

        # check for nan values of losses
        losses2 = [elem.item() for elem in losses*weight_losses]
        losses = [elem.item() for elem in losses]
        loss_per_batch.append(losses)
        gradnorm_per_batch.append(gradnorms)
        inds_per_batch.append([asdf[0] for asdf in batch.ind])

        log_n_steps = 1
        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'total {loss:>13.7} [{i+1}/{len(dataloader)}]')

        if (i % log_n_steps) == 0:
            for loss_temp, name in zip(losses2, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'(scaled loss components)')

    return loss_per_batch, gradnorm_per_batch, inds_per_batch


def train_loop_hard(dataloader, model, loss_fn1, loss_fn2, device, optimizer, weight_losses, second_order=False, scaling_factor_W=1, sobolev=False):
    # returns indices of loadcases in most difficult batches
    model.train()
    loss_per_batch = []
    inds_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)
        batch.F.requires_grad = True
        batch_size = len(batch.F)

        if sobolev:
            # Compute predictions
            pos_pred, W_pred = model(batch)
            W_pred2 = W_pred*scaling_factor_W
            # calculate stress P
            P_pred = torch.autograd.grad(W_pred2, batch.F, grad_outputs=torch.ones_like(W_pred2),create_graph=True, retain_graph=True)[0]

            # # calculate D
            # grad_outputs = torch.zeros(4, batch_size, 2, 2, device=device)
            # grad_outputs[0, :, 0, 0] = 1
            # grad_outputs[1, :, 0, 1] = 1
            # grad_outputs[2, :, 1, 0] = 1
            # grad_outputs[3, :, 1, 1] = 1
            # D_pred = torch.autograd.grad(P_pred, batch.F,
            #     grad_outputs=grad_outputs, is_grads_batched=True, retain_graph=True, create_graph=True)[0]
            # D_pred = torch.transpose(D_pred, 0, 1).reshape(-1, 2, 2, 2, 2)

            # calculate D
            grad_outputs = torch.ones(batch_size).to(device=device)
            D_pred = torch.empty(len(batch.F), 2, 2, 2, 2, device=device)
            for j in range(2):
                for k in range(2):
                    D_pred[:, j, k] = torch.autograd.grad(P_pred[:, j, k], batch.F, grad_outputs=grad_outputs, retain_graph=True,
                        create_graph=True)[0]

            # Compute losses
            losses = torch.empty(4, device=device)
            losses[0] = loss_fn1(pos_pred, batch.y)  #, batch.batch)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)
            losses[2] = loss_fn2(P_pred, batch.P)
            losses[3] = loss_fn2(D_pred, batch.D)
        else:
            pos_pred, W_pred = model(batch)
            losses = torch.empty(2, device=device)
            losses[0] = loss_fn1(pos_pred, batch.y)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)

        # Backpropagation
        loss = torch.sum(losses*weight_losses)
        optimizer.zero_grad()
        if second_order:
            # hessian needs graph for backprop
            loss.backward(create_graph=True)
        else:
            loss.backward()
        optimizer.step()

        for param in model.parameters():
            param.grad = None

        # check for nan values of losses
        losses2 = [elem.item() for elem in losses*weight_losses]
        losses = [elem.item() for elem in losses]
        if np.isnan(losses).any():
            raise ValueError('Loss')

        loss_per_batch.append(losses)
        inds_per_batch.append([asdf[0] for asdf in batch.ind])
        # print(batch.ind)

        if (i % 20) == 0:
            for loss_temp, name in zip(losses, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'total {loss:>13.7} [{i+1}/{len(dataloader)}]')

        if (i % 20) == 0:
            for loss_temp, name in zip(losses2, ['pos', 'W_scaled', 'P', 'D']):
                print(f"{name} {loss_temp:>13.7},", end=" ")
            print(f'(scaled loss components)')

    loss_per_batch = np.asarray(loss_per_batch)
    avg_loss = np.mean(loss_per_batch, axis=0)

    # calculate total loss again
    loss_per_batch = loss_per_batch*weight_losses.cpu().detach().numpy()
    loss_per_batch = np.sum(loss_per_batch, axis=1)

    # find indices of the loadcases in the worst batches
    worst_batches = np.argsort(loss_per_batch.flatten())[-2:]
    inds = [val for b in worst_batches for val in inds_per_batch[b]]
    return avg_loss, inds

def val_loop(dataloader, model, loss_fn1, loss_fn2, device,
        scaling_factor_W=1, sobolev=False):
    model.eval()
    loss_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)
        batch.F.requires_grad = True
        batch_size = len(batch.F)

        if sobolev:
            # Compute predictions
            pos_pred, W_pred = model(batch)
            W_pred2 = W_pred*scaling_factor_W
            # calculate stress P
            P_pred = torch.autograd.grad(W_pred2, batch.F, grad_outputs=torch.ones_like(W_pred2), create_graph=True)[0]

            # # calculate D
            # grad_outputs = torch.zeros(4, batch_size, 2, 2, device=device)
            # grad_outputs[0, :, 0, 0] = 1
            # grad_outputs[1, :, 0, 1] = 1
            # grad_outputs[2, :, 1, 0] = 1
            # grad_outputs[3, :, 1, 1] = 1
            #
            # D_pred = torch.autograd.grad(P_pred, batch.F,
            #     grad_outputs=grad_outputs, is_grads_batched=True)[0]
            # D_pred = torch.transpose(D_pred, 0, 1).reshape(-1, 2, 2, 2, 2)

            # calculate D
            grad_outputs = torch.ones(batch_size).to(device=device)
            D_pred = torch.empty(len(batch.F), 2, 2, 2, 2, device=device)
            for j in range(2):
                for k in range(2):
                    D_pred[:, j, k] = torch.autograd.grad(P_pred[:, j, k], batch.F, grad_outputs=grad_outputs, retain_graph=True)[0]

            # Compute losses
            losses = torch.empty(4)
            losses[0] = loss_fn1(pos_pred, batch.y)  #, batch.batch)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)
            losses[2] = loss_fn2(P_pred, batch.P)
            losses[3] = loss_fn2(D_pred, batch.D)
        else:
            pos_pred, W_pred = model(batch)
            losses = torch.empty(2, device=device)
            losses[0] = loss_fn1(pos_pred, batch.y)
            losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)

        # check for nan values of losses
        losses = [elem.item() for elem in losses]
        loss_per_batch.append(losses)
        if np.isnan(losses).any():
            raise ValueError('Loss is nan')

    loss_per_batch = np.asarray(loss_per_batch)
    avg_loss = np.mean(loss_per_batch, axis=0)

    return avg_loss

def val_loop_stress(dataloader, model, loss_fn1, loss_fn2, loss_fn3, device,
        scaling_factor_W=1, scaling_factor_P=1):
    model.eval()
    loss_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)

        pos_pred, W_pred, P_pred = model(batch)
        losses = torch.empty(3, device=device)
        losses[0] = loss_fn1(pos_pred, batch.y)
        losses[1] = loss_fn2(W_pred[:, 0], batch.W/scaling_factor_W)
        losses[2] = loss_fn3(P_pred, batch.P/scaling_factor_P)

        # check for nan values of losses
        losses = [elem.item() for elem in losses]
        loss_per_batch.append(losses)
        if np.isnan(losses).any():
            raise ValueError('Loss is nan')

    loss_per_batch = np.asarray(loss_per_batch)
    avg_loss = np.mean(loss_per_batch, axis=0)

    return avg_loss

def val_loop_stress_stiffness(dataloader, model, device, weight_losses, loss_funcs=None, scaling_factors=[1.0, 1.0, 1.0, 1.0]):
    if loss_funcs is None:
        loss_funcs = [torch.nn.MSELoss()]*4
    model.eval()
    weighted_loss_per_batch = []

    for i, batch in enumerate(dataloader):
        batch.to(device=device)

        pos_pred, W_pred, P_pred, D_pred = model(batch)
        pos_target = batch.y/scaling_factors[0]

        # per graph, scale pos back by scale_factor
        if hasattr(batch, 'scale_factor'):
            pos_pred /= batch.scale_factor[batch.batch].reshape(-1, 1)
            pos_target /= batch.scale_factor[batch.batch].reshape(-1, 1)

        losses = torch.empty(4, device=device)
        losses[0] = loss_funcs[0](pos_pred, pos_target)
        losses[1] = loss_funcs[1](W_pred[:, 0], batch.W/scaling_factors[1])
        losses[2] = loss_funcs[2](P_pred, batch.P/scaling_factors[2])
        losses[3] = loss_funcs[3](D_pred, batch.D/scaling_factors[3])

        weighted_losses = losses*weight_losses

        # check for nan values of losses
        MSEs = [elem.item() for elem in losses*scaling_factors**2]
        weighted_losses = [elem.item() for elem in weighted_losses]
        weighted_loss_per_batch.append(weighted_losses)
        if np.isnan(weighted_losses).any():
            raise ValueError('Loss')

    weighted_loss_per_batch = np.asarray(weighted_loss_per_batch)
    avg_loss = np.mean(weighted_loss_per_batch, axis=0)
    avg_MSE = avg_loss/weight_losses.cpu().numpy()*scaling_factors.cpu().numpy()**2

    return avg_loss, avg_MSE

def MSELoss_target0(pred, target, batch):
    """MSE loss applied only to the first of the possible targets

    Parameters
    ----------
    pred : torch tensor
        prediction
    target : torch tensor
        all possible targets
    batch : torch tensor
        batch that each node belongs to (is ignored)

    Returns
    -------
    torch tensor
        MSE loss for only the first target
    """
    return torch.nn.MSELoss()(pred, target[..., 0])

def MSELoss_allTargets(pred, target, batch):
    """MSE loss applied only all possible targets, then taking the minimum per graph

    Parameters
    ----------
    pred : torch tensor
        prediction
    target : torch tensor
        all possible targets
    batch : torch tensor
        graph that each node belongs to

    Returns
    -------
    torch tensor
        MSE loss for only the first target
    """
    # sum square error per graph
    SE = tg.nn.global_add_pool((pred.unsqueeze(-1) - target)**2, batch=batch.batch)
    # take the minimum error per graph
    SE = torch.min(torch.sum(SE, axis=1), axis=-1).values

    # take mean over entire batch
    MSE = torch.sum(SE)/(pred.numel())
    return MSE
