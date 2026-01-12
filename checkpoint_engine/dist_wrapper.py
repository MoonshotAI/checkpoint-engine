import torch.distributed as torch_dist


dist = torch_dist

def setup_dist():
    global dist
    import checkpoint_engine.distributed as cust_dist
    dist = cust_dist
