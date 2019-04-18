import torch


def get_batch_of_device(batch,device):
    if type(batch) is dict:
        return {k:v.to(device) for k,v in batch.items()}
    elif isinstance(batch,torch.Tensor):
        return batch.to(device)
    else:
        assert False

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device