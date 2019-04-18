import torch


def get_batch_of_device(batch,device):
    if type(batch) is dict:
        return {k:v.to(device) for k,v in batch.items()}
    elif isinstance(batch,torch.Tensor):
        return batch.to(device)
    else:
        assert False

def get_device():
    if torch.cuda.is_available():
        print('using cuda device')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')
    return device