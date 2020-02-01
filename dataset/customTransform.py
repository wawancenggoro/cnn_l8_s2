import torch
from torchvision import transforms
import numpy as np

class NormalizeL8(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        # embed()
        tensor = tensor.type(torch.FloatTensor).sub_(mean[:, None, None]).div_(std[:, None, None])

        return tensor

class DenormalizeL8(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        # embed()
        tensor = tensor.type(torch.FloatTensor).mul_(std[:, None, None]).add_(mean[:, None, None])
        return tensor
        

class NormalizeS2(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor_list):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor_list[0].device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor_list[0].device)
        # embed()
        tensor_list = [
            tensor_list[0].type(torch.FloatTensor).sub_(mean[0]).div_(std[0]),
            tensor_list[1].type(torch.FloatTensor).sub_(mean[1]).div_(std[1]),
            tensor_list[2].type(torch.FloatTensor).sub_(mean[2]).div_(std[2]),
            tensor_list[3].type(torch.FloatTensor).sub_(mean[3]).div_(std[3]),
            tensor_list[4].type(torch.FloatTensor).sub_(mean[4]).div_(std[4]),
            tensor_list[5].type(torch.FloatTensor).sub_(mean[5]).div_(std[5]),
            tensor_list[6].type(torch.FloatTensor).sub_(mean[6]).div_(std[6]),
            tensor_list[7].type(torch.FloatTensor).sub_(mean[7]).div_(std[7]),
            tensor_list[8].type(torch.FloatTensor).sub_(mean[8]).div_(std[8]),
            tensor_list[9].type(torch.FloatTensor).sub_(mean[9]).div_(std[9]),
            tensor_list[10].type(torch.FloatTensor).sub_(mean[10]).div_(std[10]),
            tensor_list[11].type(torch.FloatTensor).sub_(mean[11]).div_(std[11]),
            tensor_list[12].type(torch.FloatTensor).sub_(mean[12]).div_(std[12]),
            ]

        return tensor_list

class DenormalizeS2(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor_list):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor_list[0].device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor_list[0].device)
        # embed()
        tensor_list = [
            tensor_list[0].type(torch.FloatTensor).mul_(std[0]).add_(mean[0]).data.numpy(),
            tensor_list[1].type(torch.FloatTensor).mul_(std[1]).add_(mean[1]).data.numpy(),
            tensor_list[2].type(torch.FloatTensor).mul_(std[2]).add_(mean[2]).data.numpy(),
            tensor_list[3].type(torch.FloatTensor).mul_(std[3]).add_(mean[3]).data.numpy(),
            tensor_list[4].type(torch.FloatTensor).mul_(std[4]).add_(mean[4]).data.numpy(),
            tensor_list[5].type(torch.FloatTensor).mul_(std[5]).add_(mean[5]).data.numpy(),
            tensor_list[6].type(torch.FloatTensor).mul_(std[6]).add_(mean[6]).data.numpy(),
            tensor_list[7].type(torch.FloatTensor).mul_(std[7]).add_(mean[7]).data.numpy(),
            tensor_list[8].type(torch.FloatTensor).mul_(std[8]).add_(mean[8]).data.numpy(),
            tensor_list[9].type(torch.FloatTensor).mul_(std[9]).add_(mean[9]).data.numpy(),
            tensor_list[10].type(torch.FloatTensor).mul_(std[10]).add_(mean[10]).data.numpy(),
            tensor_list[11].type(torch.FloatTensor).mul_(std[11]).add_(mean[11]).data.numpy(),
            tensor_list[12].type(torch.FloatTensor).mul_(std[12]).add_(mean[12]).data.numpy(),
            ]

        return tensor_list

class ConvertS2WithoutDenormalization(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor_list):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        tensor_list = [
            tensor_list[0].type(torch.FloatTensor).data.numpy(),
            tensor_list[1].type(torch.FloatTensor).data.numpy(),
            tensor_list[2].type(torch.FloatTensor).data.numpy(),
            tensor_list[3].type(torch.FloatTensor).data.numpy(),
            tensor_list[4].type(torch.FloatTensor).data.numpy(),
            tensor_list[5].type(torch.FloatTensor).data.numpy(),
            tensor_list[6].type(torch.FloatTensor).data.numpy(),
            tensor_list[7].type(torch.FloatTensor).data.numpy(),
            tensor_list[8].type(torch.FloatTensor).data.numpy(),
            tensor_list[9].type(torch.FloatTensor).data.numpy(),
            tensor_list[10].type(torch.FloatTensor).data.numpy(),
            tensor_list[11].type(torch.FloatTensor).data.numpy(),
            tensor_list[12].type(torch.FloatTensor).data.numpy(),
            ]

        return tensor_list