import torch.distributed as dist        
def sync_data(input_data, rank, world_size):
    if rank == 0:
        # Rank 0 is sending it's own weight
        # to all it's siblings (1 to world_size)
        for sibling in range(1, world_size):
            dist.send(input_data, dst=sibling)
    else:
        # Siblings must recieve the parameters
        dist.recv(input_data, src=0)
