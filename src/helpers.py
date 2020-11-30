import numpy as np
from numpy.fft import ifftshift, fft2, ifft2, fftshift


def pad(array, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros([array.shape[0], int(array.shape[1] + (2 * offsets[0])), int(array.shape[2] + (2 * offsets[1]))], dtype=array.dtype)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [
        slice(None), 
        slice(int(offsets[0]), int(offsets[0] + array.shape[1])),
        slice(int(offsets[1]), int(offsets[1] + array.shape[2]))
    ]
    # Insert the array in the result at the specified offsets
    result[tuple(insertHere)] = array
    return result


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max
    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).
    '''
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse)


def complex2real(x):
    '''
    Parameter
    ---------
    x: ndarray
        assumes at least 2d. Last 2D axes are split in terms of real and imag
        2d/3d/4d complex valued tensor (n, nx, ny) or (n, nx, ny, nt)
    Returns
    -------
    y: 4d tensor (n, 2, nx, ny)
    '''
    x_real = np.real(x)
    x_imag = np.imag(x)
    y = np.array([x_real, x_imag]).astype(np.float)
    # re-order in convenient order
    if x.ndim >= 3:
        y = y.swapaxes(0, 1)
    return y


def real2complex(x):
    '''
    Converts from array of the form ([n, ]2, nx, ny[, nt]) to ([n, ]nx, ny[, nt])
    '''
    x = np.asarray(x)
    if x.shape[0] == 2 and x.shape[1] != 2:  # Hacky check
        return x[0] + x[1] * 1j
    elif x.shape[1] == 2:
        y = x[:, 0] + x[:, 1] * 1j
        return y
    else:
        raise ValueError('Invalid dimension')


def format_data(data, mask=False):
    if mask: 
        data = data * (1+1j)
    data = complex2real(data)
    return data.squeeze(0)

def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def back_format(data, mask=True):
    if mask:
        data = mask_r2c(data)
    else:
        data = real2complex(data)
    return data


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def fft2c(x, norm='ortho'):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2(ifftshift(x, axes=axes), norm=norm), axes=axes)
    return res


def ifft2c(x, norm='ortho'):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm=norm), axes=axes)
    return res



