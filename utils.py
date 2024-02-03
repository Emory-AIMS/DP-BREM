import torch
import torch.nn as nn
from opt_einsum.contract import contract

def zeros_list_like(model, device, add_dim=None):
    lst = []
    for param in model.parameters():
        size = param.data.size()
        if add_dim:
            size = (add_dim, ) + size
        lst.append( torch.zeros(size).to(device) )
    return lst


def get_list_norm(lst):
    squared_sum = 0
    for param in lst: 
        squared_sum += torch.linalg.norm(param)**2
    return torch.sqrt(squared_sum).item()

def get_list_diff_norm(lst1, lst2):
    squared_sum = 0
    for param1, param2 in zip(lst1, lst2): 
        squared_sum += torch.linalg.norm( param1-param2)**2
    return torch.sqrt(squared_sum).item()


def clip_list_(lst, max_norm):
    """
    in-place clipping
    """
    if max_norm <= 0:
        print("the 'max_norm' MUST be positive")
        raise NotImplementedError

    norm = get_list_norm(lst)
    if norm > max_norm:
        for param in lst:
            param.copy_( (max_norm/norm) * param )
    return 



def get_clip_factor_for_one_batch(lst: list, max_norm: float) -> torch.Tensor:
    batch_size = lst[0].shape[0]

    if max_norm < 0:
        clip_factor = torch.ones(batch_size)

    else:
        squared_norm = torch.zeros(batch_size)
        for param in lst:
            # check the batch size
            if param.shape[0] != batch_size:
                raise NotImplementedError("The batch size should be the same for all param! Details: {} is not equal to {}".format(batch_size, param.shape[0]))
            for i in range(batch_size):
                    squared_norm[i] += (torch.linalg.norm(param[i])**2).item()
        
        clip_factor = torch.clamp(max_norm/torch.sqrt(squared_norm), max=1)

    return clip_factor


def _get_smoothed_and_clipped_gradient_inefficient(model, data, target, args, device, max_record_grad_norm=-1):
    
    # For a batch of training data, compute the average of all per-example gradients, where each per-example gradient
    # is smoothed and then clipped
    
    criterion = nn.CrossEntropyLoss()

    noise_smooth = args.noise_smooth
    batch_size = len(target)
    model.train()
    
    if noise_smooth == 0 and max_record_grad_norm < 0: # which means no-smooth and no-record-clip
        grad = zeros_list_like(model, device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for idx, param in enumerate(model.parameters()):
            grad[idx].add_( param.grad )
        return grad

    
    
    def update_grad_sample(model, grad_sample):
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for idx, param in enumerate(model.parameters()):
            grad_sample[idx].add_( param.grad_sample )

    

    grad_sample = zeros_list_like(model, device, add_dim=batch_size) 
    if noise_smooth == 0:
        update_grad_sample(model, grad_sample)
    else:
        num_smooth = args.num_smooth
        state_dict_copy = model.state_dict().copy()
        for _smooth_idx in range(num_smooth):
            model.load_state_dict(state_dict_copy, strict=True)
            for param in model.parameters():
                param.data.add_(torch.normal(0, noise_smooth, param.shape).to(device))
            update_grad_sample(model, grad_sample)
        model.load_state_dict(state_dict_copy, strict=True)
    
        # average grad_sample over multiple smoothings
        for param in grad_sample:
            param.copy_( param/num_smooth )
    
    
    # clip the grad per-example and accumulate to "grad"
    clip_factor = get_clip_factor_for_one_batch(grad_sample, max_record_grad_norm)
    grad = zeros_list_like(model, device)
    for idx, param in enumerate(grad_sample):
        for i in range(batch_size):
            grad[idx].add_( param[i]*clip_factor[i]/batch_size )    # devided by batch_size to get the averaged value
    
    return grad




def get_smoothed_and_clipped_gradient(model, data, target, args, device, max_record_grad_norm=-1):
    """
    For a batch of training data, compute the average of all per-example gradients, where each per-example gradient
    is smoothed and then clipped
    """

    if args.use_inefficient_clip:
        return _get_smoothed_and_clipped_gradient_inefficient(model, data, target, args, device, max_record_grad_norm)


    criterion = nn.CrossEntropyLoss()

    noise_smooth = args.noise_smooth
    batch_size = len(target)
    model.train()
    
    if noise_smooth == 0 and max_record_grad_norm < 0: # which means no-smooth and no-record-clip
        grad = zeros_list_like(model, device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for idx, param in enumerate(model.parameters()):
            grad[idx].add_( param.grad )
        return grad

    
    

    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Reference: https://opacus.ai/api/_modules/opacus/optimizers/optimizer.html#DPOptimizer
    
    # a flat list of per sample gradient tensors (one per parameter)
    grad_samples = []
    for p in model.parameters():
        grad_samples.append(_get_flat_grad_sample(p))


    # Performs gradient clipping.
    if len(grad_samples[0]) == 0:
        per_sample_clip_factor = torch.zeros((0,))
    else:
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in grad_samples]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (max_record_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

    grad = []
    for p in model.parameters():
        grad_sample = _get_flat_grad_sample(p)
        grad_for_p = contract("i,i...", per_sample_clip_factor, grad_sample)
        grad.append(grad_for_p/batch_size)

    
    return grad


def _get_flat_grad_sample(p: torch.Tensor):
        """
        Return parameter's per sample gradients as a single tensor.

        By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
        batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
        only one batch, and a list of tensors if gradients are accumulated over multiple
        steps. This is done to provide visibility into which sample belongs to which batch,
        and how many batches have been processed.

        This method returns per sample gradients as a single concatenated tensor, regardless
        of how many batches have been accumulated

        Args:
            p: Parameter tensor. Must have ``grad_sample`` attribute

        Returns:
            ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
            concatenating every tensor in ``p.grad_sample`` if it's a list

        Raises:
            ValueError
                If ``p`` is missing ``grad_sample`` attribute
        """

        if not hasattr(p, "grad_sample"):
            raise ValueError(
                "Per sample gradient not found. Are you using GradSampleModule?"
            )
        if p.grad_sample is None:
            raise ValueError(
                "Per sample gradient is not initialized. Not updated in backward pass?"
            )
        if isinstance(p.grad_sample, torch.Tensor):
            ret = p.grad_sample
        elif isinstance(p.grad_sample, list):
            ret = torch.cat(p.grad_sample, dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        return ret


def get_clipped_avg(global_momentum, selected_client_momentums, max_client_clip_norm, args, device):
    # initialize the difference for client and the sum 
    clipped_avg = [torch.zeros_like(param, device=device) for param in global_momentum]
    clipped_one = [torch.zeros_like(param, device=device) for param in global_momentum]
    num_selected_clients = len(selected_client_momentums)
    for client_momentum in selected_client_momentums:

        # determine the clipping objective
        for idx, param in enumerate(clipped_one):
            if args.client_clip_type == "center_clip":
                param.copy_(client_momentum[idx] - global_momentum[idx])
            else:
                param.copy_(client_momentum[idx])
                

        # clipping (if necessary)
        if max_client_clip_norm > 0:
            clip_list_(clipped_one, max_client_clip_norm)

        # add the clipped one to the sum
        for idx, param in enumerate(clipped_avg):
            param.add_(clipped_one[idx]/num_selected_clients)

    return clipped_avg


