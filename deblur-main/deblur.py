"""
Implementation of 'High-Quality Motion Deblurring from a Single Image (SIGGRAPH 2008)'

    Eduardo Stuani
    Gustavo Prolla Lacroix

    NOTE:
    * Works with numba version 0.58.1
    * Each channel of the image is processed independently, with the exception
      of the local prior mask M, which is calculated using the standard
      deviation of all channels.
"""
import cv2
import warnings
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import lsq_linear, nnls
from numba import njit, jit
from numba.core.errors import NumbaPendingDeprecationWarning
from convolve import psf2otf, toeplitz_transform, matrix_to_vector, vector_to_matrix, expand_matrix, expand_vector, extract_rows_top_sd
from helpers import open_image, write_image, kernel_from_image


# Filter Numba deprecation warnings
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

def get_derivatives(matrix):
    """
    Calculates all derivatives of matrix from the set theta 
    of derivative operators.

    Parameters:
    - matrix: 2D array

    Returns
    - derivatives: dictionary containg the derivatives
    """
    derivatives = {
        'd0': matrix.copy(),
        'dx': np.gradient(matrix, axis=1),
        'dy': np.gradient(matrix, axis=0),
        'dxx': np.gradient(np.gradient(matrix, axis=1), axis=1),
        'dyy': np.gradient(np.gradient(matrix, axis=0), axis=0),
        'dxy': np.gradient(np.gradient(matrix, axis=1), axis=0),
        'dyx': np.gradient(np.gradient(matrix, axis=0), axis=1),
    }
    return derivatives

def computeLocalPrior(I, f, t):
    """
    Compute the local prior M for each pixel in I. The standard deviation
    is calculated for each channel. All channels must be below the threshold
    t for the pixel to be considered in the local prior.

    Parameters:
    - I: 2D array, the input image (observed image) with shape (height, width, channels)
    - f: 2D array, the PSF or filter kernel
    - t: float, threshold for the standard deviation

    Returns:
    - M: 2D array, the local prior mask with shape (height, width)
    """
    I = np.atleast_3d(I)
    M = np.zeros(I.shape[:2], dtype=np.uint8)
    std_dev = np.zeros(1)

    # Iterate through pixels
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            # Extract local window centered at (i, j) with size defined by f
            top = max(0, i - f[0] // 2)
            bottom = min(I.shape[0], i + f[0] // 2 + 1)
            left = max(0, j - f[1] // 2)
            right = min(I.shape[1], j + f[1] // 2 + 1)
            window = I[top:bottom, left:right]

            cv2.meanStdDev(window, None, std_dev)
            M[i, j] = std_dev[0] < t
    return M

def save_mask_as_image(mask, output_path):
    """
    Saves a mask as an image.

    Parameters:
    - mask: 2D array, the mask to be saved
    - output_path: string, the path to save the image
    """
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)

@njit
def updatePsi(I_d, L_d, M, lambda1, lambda2, gamma):
    """
    Updates the Psi values for a single channel. By the paper definition,
    Psi ~ L_d. Therefore, gamma is used to weight the latent image gradient
    and is increased at each iteration.

    Parameters:
    - I_d: 3D array, the observed image gradient with shape (direction, height, width)
    - L_d: 3D array, the latent image gradient with shape (direction, height, width)
    - M: 2D array, the local prior mask with shape (height, width)
    - lambda1: float, the weight for the global prior
    - lambda2: float, the weight for the local prior
    - gamma: float, the weight for the latent image gradient

    Returns:
    - nPsi: 3D array, the updated Psi with shape (direction, height, width)
    """
    k = 2.7
    a = 6.1e-4
    b = 5.0
    lt = 1.852
    x = np.zeros(3)
    nPsi = [np.zeros_like(M), np.zeros_like(M)]
    for v in range(2):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                x[0] = (lambda2*M[i, j]*I_d[v][i, j] + gamma*L_d[v][i, j])/(-a*lambda1 + lambda2*M[i, j] + gamma)
                x[1] = ((k/2)*lambda1 + lambda2*M[i, j]*I_d[v][i, j] + gamma*L_d[v][i, j])/(lambda2*M[i, j] + gamma)
                x[2] = ((-k/2)*lambda1 + lambda2*M[i, j]*I_d[v][i, j] + gamma*L_d[v][i, j])/(lambda2*M[i, j] + gamma)
                result = np.min(x)
                nPsi[v][i, j] = result 
    return nPsi

def computeL(L, I, f, Psi, gamma):
    """
    Compute the latent image L for a single channel.

    Parameters:
    - L: 2D array, the latent image with shape (height, width)
    - I: 2D array, the observed image with shape (height, width)
    - f: 2D array, the PSF or filter kernel
    - Psi: 3D array, the latent image gradient with shape (direction, height, width)
    - gamma: float, the weight for the latent image gradient

    Returns:
    - L_star: 2D array, the updated latent image with shape (height, width)
    """
    # Derivatives and derivative weights for the Delta calculation
    d = get_derivatives(L)
    d_w = {
        'd0': 0,
        'dx': 1,
        'dy': 1,
        'dxx': 2,
        'dyy': 2,
        'dxy': 2,
        'dyx': 2
    }

    # Calculate Delta
    delta = np.zeros(I.shape, np.complex128)
    for key in d:
        dv = fft2(d[key])
        delta += np.complex128(50/(2**d_w[key]))*np.conjugate(dv)*dv

    # Calculate L*
    F_f = psf2otf(f, I.shape)
    CF_f = np.conjugate(F_f)
    F_dx = fft2(d['dx'])
    CF_dx = np.conjugate(F_dx)
    F_dy = fft2(d['dy'])
    CF_dy = np.conjugate(F_dy)
    L_nominator = CF_f*fft2(I)*delta + gamma*(CF_dx*fft2(Psi[0])) + gamma*(CF_dy*fft2(Psi[1]))
    L_denominator = CF_f*F_f*delta + gamma*(CF_dx*F_dx) + gamma*(CF_dy*F_dy)
    L_star = L_nominator / L_denominator
    L_star = ifft2(L_star).real.astype(np.float64)
    return L_star

def updatef(L, I, f, n_rows=50, k_cut_ratio=1e-5):
    """
    Fix L and I and update the blur kernel f.

    Parameters:
    - L: 3D array, the latent image with shape (height, width, channels)
    - I: 3D array, the observed image with shape (height, width, channels)
    - f: 2D array, the PSF or filter kernel
    # TODO
    Returns:
    - optimized_f : 2D array, the updated kernel f with shape (height, width)
    """

    # Derivatives and derivative weights for the Theta calculation
    dL = get_derivatives(L)
    dI = get_derivatives(I)
    d_w = {
        'd0': 0,
        'dx': 1,
        'dy': 1,
        'dxx': 2,
        'dyy': 2,
        'dxy': 2,
        'dyx': 2
    }

    # Calculate the theta for I_w
    I_w = np.zeros(I.shape, np.float64)
    for key, weight in d_w.items():
        I_w += weight * dI[key] 

    # Calculate the theta for L_w
    L_w = np.zeros(L.shape, np.float64)
    for key, weight in d_w.items():
        L_w += weight * dL[key] 

    write_image(f'results/iw.png', I_w.copy())
    write_image(f'results/lw.png', L_w.copy())


    # Extract a piece of the image (may not be enough space to store the entire matrix in the memory after)
    # Use de mean of the three channels
    sel_L_w = np.mean(L_w, axis=2)[n_rows:-n_rows, n_rows:-n_rows]
    sel_I_w = np.mean(I_w, axis=2)[n_rows:-n_rows, n_rows:-n_rows]
    
    # Get A transforming the selected latent image into a toeplitz matrix
    A = toeplitz_transform(sel_L_w, f)

    # Get B
    B_row_num = sel_L_w.shape[0] + f.shape[0] - 1
    B_col_num = sel_L_w.shape[1] + f.shape[1] - 1
    B = expand_matrix(sel_I_w, (B_row_num, B_col_num))
    B = matrix_to_vector(B)

    # Minimize the problem, obs: this is our heart, but is not beating 😔
    result_l2 = lsq_linear(A, B, method='trf', bounds=(0, np.inf), lsmr_tol=1e-5, verbose=0)
    optimized_f = vector_to_matrix(result_l2.x, f.shape) 
    optimized_f[optimized_f < k_cut_ratio] = 0 
    total_sum = np.sum(optimized_f)
    if total_sum != 0:
        optimized_f = optimized_f / total_sum 
    else:
        optimized_f = optimized_f 
    return optimized_f  
    
