import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d
from scipy.linalg import toeplitz


def create_line_psf(theta, scale, sz):
    """
    Create a Point Spread Function (PSF) in the form of a straight line.

    Parameters:
    - theta: angle of the line in radians
    - scale: float, in the interval [0, 1]
    - sz: tuple, the desired size of the PSF

    Returns:
    - psf: 2D array, the PSF

    TODO:
    - remove division by zero
    """
    psf = np.zeros(sz)
    X = sz[1] // 2
    Y = sz[0] // 2
    theta = theta % np.pi

    # Calculate intersection points
    if theta <= np.pi/2:
        p1 = (min(Y/np.tan(theta), X), min(X*np.tan(theta), Y))
    else:
        p1 = (max(Y/np.tan(theta), -X), min(-X*np.tan(theta), Y))

    # Calculate scaled points
    p1 = (int(p1[0]*scale), int(p1[1]*scale))
    p2 = (-p1[0], -p1[1])

    # Change coordinate system
    p1 = (p1[0]+X, -p1[1]+Y)
    p2 = (p2[0]+X, -p2[1]+Y)

    # Draw line
    psf = cv2.line(psf, p2, p1, color=1, thickness=1)
    # Normalize
    psf = psf / np.sum(psf)
    return psf

def create_sparse_psf(num_points, sz):
    """
    Create a Point Spread Function (PSF) with num_points spaced randomly in space.

    Parameters:
    - num_points: number of points
    - sz: tuple, the desired size of the PSF

    Returns:
    - psf: 2D array, the PSF

    """
    psf = np.zeros(sz)
    
    # make random choices for the sparse points
    points_x = np.random.randint(0, sz[1], num_points)
    points_y = np.random.randint(0, sz[0], num_points)
    
    # define the sparse points in the kernel
    for y, x in zip(points_x, points_y):
        psf[y, x] = 1.0
    
    return psf / np.sum(psf)  

def psf2otf(psf, sz):
    """
    Compute the FFT of the Point Spread Function (PSF) and pad/crop it so
    it has the shape specified by sz.

    Parameters:
    - psf: 2D array, the Point Spread Function.
    - sz: tuple, the desired size of the output OTF.

    Returns:
    - otf: 2D array, the Optical Transfer Function.
    """
    psf = np.atleast_2d(psf)
    psf_sz = psf.shape

    # Pad/Crop in x
    diffx = np.abs(sz[0] - psf_sz[0])
    diffx2 = diffx // 2
    restx = diffx % 2
    if psf_sz[0] > sz[0]:
        psf = psf[diffx2:-(diffx2+restx), :]
    elif psf_sz[0] < sz[0]:
        psf = np.pad(psf, [(diffx2, diffx2+restx), (0, 0)], mode='constant')    

    # Pad/Crop in y
    diffy = np.abs(sz[1] - psf_sz[1])
    diffy2 = diffy // 2
    resty = diffy % 2
    if psf_sz[1] > sz[1]:
        psf = psf[:, diffy2:-(diffy2+resty)]
    elif psf_sz[1] < sz[1]:
        psf = np.pad(psf, [(0, 0), (diffy2, diffy2+resty)], mode='constant')    


    otf = fft2(fftshift(psf))
    return otf

def convolve_in_frequency_domain(image, psf):
    """
    Convolve an image with a point spread function (PSF) in the frequency domain.

    Parameters:
    - image: 2D/3D array, the input image with shape (height, width, channels).
    - psf: 2D array, the point spread function.

    Returns:
    - convolved_image: 3D array, the convolved image with shape (height, width, channels).

    NOTE:
    Will add option for padding in the future. The problem with padding is that
    you can't simply deconvolve the image with the same kernel and restore it
    """
    psf = np.atleast_2d(psf)
    image = np.atleast_3d(image)
    num_channels = image.shape[2]
    convolved_image = np.zeros_like(image, dtype=np.complex128)
    otf_shape = (image.shape[0], image.shape[1])
    otf = psf2otf(psf, otf_shape)

    # Perform convolution in the frequency domain for each channel
    for channel in range(num_channels):
        image_fft = fft2(image[:, :, channel])
        result_fft = image_fft * otf
        convolved_image[:, :, channel] = np.real(ifft2(result_fft))
    
    return convolved_image

def convolve_in_spatial_domain(image, kernel):
    """
    Convolve a 3-channel image with a kernel in the spatial domain.

    Parameters:
    - image: 3D array, the input image with shape (height, width, channels)
    - kernel: 2D array, the kernel or PSF

    Returns:
    - convolved_image: 3D array, the convolved image with same orginal shape
    """
    image = np.atleast_3d(image)
    convolved_image = np.zeros_like(image)
    for v in range(image.shape[2]):
        convolved_image[:, :, v] = convolve2d(image[:, :, v], kernel, mode='same', boundary='symm')
    return convolved_image

def deconvolve(image, psf):
    """
    Deconvolve a 3-channel image with a point spread function (PSF)

    Parameters:
    - image: 3D array, the input image with shape (height, width, channels).
    - psf: 2D array, the point spread function.

    Returns:
    - deconvolved_image: 3D array, the deconvolved image with shape (height, width, channels).

    NOTE:
    See `convolve_in_frequency_domain` for notes about padding
    """
    num_channels = image.shape[2]
    deconvolved_image = np.zeros_like(image, dtype=np.complex128)
    otf_shape = (image.shape[0], image.shape[1])
    otf = psf2otf(psf, otf_shape)

    # Perform deconvolution in the frequency domain for each channel
    for channel in range(num_channels):
        image_fft = fft2(image[:, :, channel])
        result_fft = image_fft / otf
        deconvolved_image[:, :, channel] = np.real(ifft2(result_fft))
    
    return deconvolved_image

def add_gaussian_noise(image, mean=0, std=1):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image: 2D NumPy array (grayscale) or 3D NumPy array (RGB).
    - mean: Mean of the Gaussian distribution (default is 0).
    - std: Standard deviation of the Gaussian distribution (default is 1).

    Returns:
    - Noisy image with added Gaussian noise.
    """

    # Generate Gaussian noise with the same shape as the input image
    noise = np.random.normal(loc=mean, scale=std, size=image.shape)

    # Add the noise to the image
    noisy_image = image + noise

    # Clip the values to the valid range [0, 255] for uint8 images
    noisy_image = np.clip(noisy_image, 0, 255)

    # Round to integers if the image is of integer type
    if np.issubdtype(image.dtype, np.integer):
        noisy_image = np.round(noisy_image).astype(image.dtype)

    return noisy_image

def toeplitz_transform(L, f, print_ir=False):
    """
    Transform the Latent Image into a doubly blocked toeplitz matrix.
    The kernel is used to determine the number of columns   
    Parameters:
    # TODO
    - L: Latent Image, numpy 2D matrix ()
    - f: Kernel, 2D numpy matrix ()
    - print_ir: if True, all intermediate results will be printed after each step of the algorithms

    Returns:
    doubly_blocked: doubly blocked A matrix (determined by L)
    """
    f_row_num, f_col_num = f.shape 

    # number of columns and rows of the image
    L_row_num, L_col_num = L.shape

    #  calculate the output dimensions
    output_row_num = f_row_num + L_row_num - 1
    output_col_num = f_col_num + L_col_num - 1
    if print_ir: print('output dimension:', output_row_num, output_col_num)

    # zero pad the filter
    L_zero_padded = np.pad(L, ((output_row_num - L_row_num, 0),
                               (0, output_col_num - L_col_num)),
                            'constant', constant_values=0)
    if print_ir: print('L_zero_padded: ', L_zero_padded)

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    # Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(L_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = L_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(f_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
        if print_ir: print('L '+ str(i)+'\n', toeplitz_m)

    # doubly blocked toeplitz indices: 
    # this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, L_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(f_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    if print_ir: print('doubly indices \n', doubly_indices)

    # creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    if print_ir: print('doubly_blocked: ', doubly_blocked)

    return doubly_blocked

def matrix_to_vector(input):
    """
    Converts the input matrix to a vector by stacking the rows in 
    a specific way for the optimization
    
    Parameters:
    - input: a numpy 2D matrix
    
    Returns:
    - ouput_vector: a column vector with size input.shape[0]*input.shape[1]
    """
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    # flip the input matrix up-down because last row should go first
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row   
    return output_vector

def vector_to_matrix(input, output_shape):
    """
    Reshapes the output of the matrix multiplication (the convolution) 
    to the shape "output_shape"
    
    Parameters:
    input -- a numpy vector
    
    Returns:
    output -- numpy matrix with shape "output_shape"
    """
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # flip the output matrix up-down to get correct result
    output=np.flipud(output)
    return output

def expand_vector(vector, new_size):
    """
    Pads the given vector with zeros to increase its size to a specified new size 
    while maintaining the original data in the center
    
    Parameters:
    - vector: a 1D numpy vector
    
    Returns:
    - new_size: a 1D numpy vector with size "new_size"
    """

    current_size = len(vector)
    missing_zeros = new_size - current_size
    half_zeros = missing_zeros // 2

    new_vector = np.zeros(new_size, dtype=vector.dtype)
    new_vector[half_zeros:current_size + half_zeros] = vector

    return new_vector

def expand_matrix(matrix, new_shape):
    """
    Pads the given matrix with zeros to increase its size to a specified new size 
    while maintaining the original data in the center

    Parameters:
    - vector: a 2D numpy matrix
    
    Returns:
    - new_size: a 2D numpy matrix with shape "new_shape"
    """
    current_shape = matrix.shape
    diff_rows = new_shape[0] - current_shape[0]
    diff_cols = new_shape[1] - current_shape[1]

    top_pad = diff_rows // 2
    bottom_pad = diff_rows - top_pad
    left_pad = diff_cols // 2
    right_pad = diff_cols - left_pad

    expanded_matrix = np.pad(matrix, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
    return expanded_matrix

def extract_rows_top_sd(L, percentage, I):
    """
    Select rows of L with the largest standard deviations for a specific color channel.
    TODO: this sucks (not the method, the result ...) the method sucks too

    Parameters:
    - L: Latent image (matrix) 2d.
    - percentage: Percentage of rows to be selected.

    Returns:
    - selected_L: Latent image containing only the selected rows.
    """
    
    # Extract the specified color channel
    # Calculate the standard deviation along axis 1 (rows) for the specified color channel
    std_devs = np.std(L, axis=1)

    # Calculate the number of rows to be selected based on the percentage
    num_rows = min(int(percentage * L.shape[0]), L.shape[0])
    
    # Get the indices that would sort the elements in descending order
    selected_indices = np.argsort(std_devs)[-num_rows:]

    # Extract the selected rows from the latent image
    selected_L = L[selected_indices, :]
    selected_I = I[selected_indices, :]

    return selected_L, selected_I