import torch
from model.compress import Transform, AdaptiveDWT
is_test_Transform = False
is_test_AdaptiveDWT = False

if __name__ == '__main__':
    if is_test_Transform:
        torch.set_default_dtype(torch.float64)

        x = torch.Tensor(10, 64, 8, 8).cuda()
        x.uniform_(0, 255)

        transform = Transform(64).cuda()
        transform.update()

        x_tran = transform(x, is_encoder=True)
        x_reconstruct = transform(x_tran, is_encoder=False)

        error = torch.abs(x - x_reconstruct)
        assert error.max().item() < 1e-10, (error.mean(), error.max())

    if is_test_AdaptiveDWT:
        """Disable round of DWT before test"""
        torch.set_default_dtype(torch.float64)

        x_size = 32
        x = torch.Tensor(4, 4, x_size, x_size).cuda()
        x.uniform_(0, 255)
        q_table_dwt = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.get_default_dtype())
        adaptiveDWT = AdaptiveDWT(x_size, level=5, q_table=q_table_dwt).cuda()
        X = adaptiveDWT(x, is_encoder=True)
        x_reconstruct = adaptiveDWT(X, is_encoder=False)

        error = torch.abs(x - x_reconstruct)
        assert error.max().item() < 1e-10, (error.mean(), error.max())
