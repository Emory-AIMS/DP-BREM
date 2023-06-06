import torch
import torch.nn as nn

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
    return torch.sqrt(squared_sum)



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



def get_smoothed_and_clipped_gradient(model, data, target, args, device, max_record_grad_norm=-1):
    """
    For a batch of training data, compute the average of all per-example gradients, where each per-example gradient
    is smoothed and then clipped
    """
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





