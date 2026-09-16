import numpy as np
def qubo_brute(Q):
    '''
    code from faster QUBO brute-force solving using gray code
    https://arxiv.org/pdf/2310.19373
    NB algorithm for MAX on CPU
    '''
    N=len(Q)
    # initialize bit vector and value
    x = np.zeros(N, dtype=int)
    v = 0
    # initialize minimal bit vector and value
    x_max = np.zeros(N, dtype=int)
    v_max = 0
    places = 2 ** np.arange(N) # can be outside loop
    for k in range (1, 2** N):
        x[:] = ( k & places ) > 0 # get binary vector from k
        v = x @ Q @ x # get QUBO objective value
        if v > v_max : # check for maximality
            x_max[:] = x # memorize binary vector ..
            v_max = v # .. and value
    return x_max , v_max


'''code from https://github.com/AlexanderNenninger/QUBOBrute/blob/master/qubobrute/core.py on GPU'''
from typing import Dict, Tuple, Union
import numba as nb
import numpy as np
from numba import cuda

#Qubo = Dict[Tuple[int, int], float]


@nb.njit(fastmath=True)
def bits(n: Union[int, np.intp], nbits: int) -> np.ndarray:
    """Turn n into an array of float32.

    Args:
        n (int)
        nbits (int): length of output array

    Returns:
        The bits of n in an array of float32
    """
    bits = np.zeros(nbits, dtype=np.float32)
    i = 0
    while n > 0:
        n, rem = n // 2, n % 2
        bits[i] = rem
        i += 1
    return bits


# CUDA Code starts here
@cuda.jit(device=True)
def cu_bits(n, xs):
    i = 0
    while n > 0:
        n, rem = n // 2, n % 2
        xs[i] = rem
        i += 1


@cuda.jit(device=True)
def cu_qnorm(q, x):
    """Calculate x^T q x inside the CUDA kernel

    Args:
        q (_type_): 2d array of size nbits x nbits
        x (_type_): 1d array of size nbits

    Returns:
        float: x^T q x
    """
    n = q.shape[0]
    out = 0
    for i in range(n):
        tmp = 0
        for j in range(n):
            tmp += q[i, j] * x[j]
        out += tmp * x[i]

    return out



def solve_gpu(Q: np.ndarray, c: np.float32) -> np.ndarray:
    """Solve QUBO H(x) = x^T Q x + c on a GPU.

    Args:
        q (np.ndarray): Q
        c (np.float32): energy offset

    Returns:
        v (np.ndarray): All possible energy values H can take in enumerated order.
        Suppose  argmin(v) = i, then bits(i, q.shape[0]) minimizes H.
    """
    assert (
        Q.ndim == 2
    ), f"q needs to be a square matrix. Got {Q.ndim=}, but expected q.ndim=2."
    assert Q.shape[0] == Q.shape[1], "q needs to be a square matrix."

    nbits = Q.shape[0]
    N = 2**nbits

    @cuda.jit()
    def kernel(q, c, solutions):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x
        idx = tx + ty * bw  # type: ignore
        xs = cuda.local.array(32, dtype=nb.u1)  # type: ignore
        cu_bits(idx, xs)
        if 0 <= idx < solutions.size:
            solutions[idx] = cu_qnorm(q, xs) + c

    solutions = cuda.device_array(N, dtype=np.float16)
    threadsperblock = 256
    blockspergrid = (solutions.size + (threadsperblock - 1)) // threadsperblock

    Q = cuda.to_device(Q)
    kernel[blockspergrid, threadsperblock](Q, c, solutions)  # type: ignore

    return solutions.copy_to_host()

import torch
import time
def solve_gpu_torch(Q: np.ndarray, c: float, device: torch.device) -> np.ndarray:
    """Solve QUBO H(x) = x^T Q x + c by brute-force enumeration.

    Works on 'cuda', 'mps', or 'cpu' — same code path for all.
    """
    assert Q.ndim == 2 and Q.shape[0] == Q.shape[1], "Q must be a square matrix."

    nbits = Q.shape[0]
    N = 2 ** nbits

    Q_t = torch.as_tensor(Q, dtype=torch.float32, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    t0 = time.perf_counter()

    # generate all 2^nbits bit combinations as a (N, nbits) matrix of 0/1
    idx = torch.arange(N, device=device, dtype=torch.int64)
    bits = ((idx.unsqueeze(1) >> torch.arange(nbits, device=device)) & 1).float()
    # bits[i] is the bit vector for state i, matching your cu_bits(idx, xs) convention
    # (check bit order matches your original cu_bits — see note below)

    # compute x^T Q x for every row at once: (N, nbits) @ (nbits, nbits) -> (N, nbits), then dot with bits
    Qx = bits @ Q_t                      # (N, nbits)
    energies = (Qx * bits).sum(dim=1) + c  # (N,)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    print(f"[solve_gpu] nbits={nbits}, N=2^{nbits}={N}, device={device}, "
          f"time={elapsed*1000:.3f} ms")

    return energies.cpu().numpy()

def qubo_brute_gpu(Q, device):
    """
    Wrapper that uses the GPU via Numba and formats the output 
    to match the structure expected by the main script.
        
    Returns:
        best_x (np.array): Binary vector of the best solution found.
        best_val (float): The energy value of the best solution.
    """
    # start timer

    # Get all energies (2^N values) calculated in parallel
    all_energies = solve_gpu_torch(Q, c=0.0, device=device)
    
    # Find the winner
    # we look for the highest value (argmax).
    best_idx = np.argmax(all_energies)
    best_val = all_energies[best_idx]
    
    # Convert the winning index into a binary vector
    nbits = Q.shape[0]
    best_x = bits(best_idx, nbits).astype(int) # Use the CPU helper function
    
    return best_x, best_val