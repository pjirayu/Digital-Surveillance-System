import torch
def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    save_name = save_dir + '_{}'.format(filename)
    torch.save(state, save_name)
