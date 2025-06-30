import math
import numpy as np
import torch
from torch import Tensor


def improved_timesteps_schedule_decrease(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
    constant_steps: int = 0
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    """
    if constant_steps == 0:
        total_training_steps_prime = math.floor(
            total_training_steps
            / (math.log2(math.ceil(initial_timesteps / final_timesteps)) + 1)
        )
        num_timesteps = initial_timesteps // math.pow(
            2, math.floor(current_training_step / total_training_steps_prime)
        )
        num_timesteps = max(num_timesteps, final_timesteps) + 1
    else:
        num_timesteps = constant_steps + 1

    return num_timesteps

def improved_timesteps_schedule_decrease_linear(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
    constant_steps: int = 0
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    """
    if constant_steps == 0:
        total_training_steps_prime = math.floor(
            total_training_steps
            / (initial_timesteps - final_timesteps + 1)
        )
        num_timesteps = initial_timesteps - math.floor(
            current_training_step / total_training_steps_prime
        )
        num_timesteps = max(num_timesteps, final_timesteps) + 1
    else:
        num_timesteps = constant_steps + 1

    return num_timesteps

def improved_timesteps_schedule_increase_linear(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
    constant_steps: int = 0
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    """
    if constant_steps == 0:
        total_training_steps_prime = math.floor(
            total_training_steps
            / (final_timesteps - initial_timesteps + 1)
        )
        num_timesteps = initial_timesteps + math.floor(
            current_training_step / total_training_steps_prime
        )
        num_timesteps = min(num_timesteps, final_timesteps) + 1
    else:
        num_timesteps = constant_steps + 1

    return num_timesteps

def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
    constant_steps: int = 0
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    """
    if constant_steps == 0:
        total_training_steps_prime = math.floor(
            total_training_steps
            / (math.log2(math.floor(initial_timesteps / final_timesteps)) + 1)
        )
        num_timesteps = initial_timesteps // math.pow(
            2, math.floor(current_training_step / total_training_steps_prime)
        )
        num_timesteps = max(num_timesteps, final_timesteps) + 1
    else:
        num_timesteps = constant_steps + 1

    return num_timesteps

def improved_timesteps_schedule_increase(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
    constant_steps: int = 0
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    if constant_steps == 0:
        total_training_steps_prime = math.floor(
            total_training_steps
            / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
        )
        num_timesteps = initial_timesteps * math.pow(
            2, math.floor(current_training_step / total_training_steps_prime)
        )
        num_timesteps = min(num_timesteps, final_timesteps) + 1
    else:
        num_timesteps = constant_steps + 1

    return num_timesteps

def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas


def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps


def improved_loss_weighting(sigmas: Tensor) -> Tensor:
    """Computes the weighting for the consistency loss.

    Parameters
    ----------
    sigmas : Tensor
        Standard deviations of the noise.

    Returns
    -------
    Tensor
        Weighting for the consistency loss.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    return 1 / (sigmas[1:])


def pseudo_huber_loss(input: Tensor, target: Tensor) -> Tensor:
    """Computes the pseudo huber loss.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    target : Tensor
        Target tensor.

    Returns
    -------
    Tensor
        Pseudo huber loss.
    """
    c = 0.00054 * math.sqrt(math.prod(input.shape[1:]))
    return torch.sqrt((input - target) ** 2 + c**2) - c

def q_sample(
        y0: Tensor,
        x0: Tensor,
        sigmas: Tensor,
        alphas: Tensor,
        timestep: Tensor,
        noise: Tensor = None,
        
):
    """forward process of ResShift type diffusion.
    Parameters
    ----------
        x_start: Tensor,
        lq: Tensor,
        sigmas: Tensor,
        timesteps: Tensor,
        noise: Tensor,

    Returns
    -------
    Tensor
        x_t
    """
    e0 = y0 - x0
    sigma = sigmas[timestep].reshape([x0.shape[0], 1, 1, 1])
    alpha = alphas[timestep].reshape([x0.shape[0], 1, 1, 1])
    if noise is None:
        noise = torch.randn_like(x0)
    x_t = x0 + e0 * alpha + sigma * noise
    return x_t



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)