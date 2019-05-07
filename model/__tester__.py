import torch
from model.compress import Transform

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    x = torch.Tensor(10, 64, 8, 8).cuda()
    x.uniform_(0, 255)

    transform = Transform(64).cuda()
    transform.update()

    x_tran = transform(x, is_encoder=True)
    x_reconstruct = transform(x_tran, is_encoder=False)

    error = torch.abs(x - x_reconstruct)
    assert error.max().item() < 1e-10, (error.mean(), error.max())
