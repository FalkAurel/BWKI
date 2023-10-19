import numpy as np

def RGBToGrayScale(data, kind="rgb"):
    """
    Takes a rgb or bgr picture as input and outputs the grayscale representation. [kind] is used to specify what
    kind of transformation is needed depending on whether the picture is rgb or bgr. By default [kind] is set to
    "rgb".
    """
    if kind.lower() == "rgb":
        return data.dot(np.array([0.1140, 0.5870, 0.2989]))
    return data.dot(np.array([0.2989, 0.5870, 0.1140]))

def convertingBGR2RGB(data):
    """
    Converts bgr images into rgb images.
    """
    return data[:, :, ::-1]

def convertingToBlackAndWhite(data):
    """
    Converts rgb and gray scale images to black and white images.
    """
    if data.ndim < 3:
        return np.where(data > 127, 255, 0).astype(np.uint8)
    return np.where(data.mean(axis=-1) >= 127.5, 255, 0).astype(np.uint8)

def _gaussianKernel(length, *, sigma=1):
    """
    Creates gaussian kernel with side length [length] and a standard deviation of [sigma]. The output will be 2d.
    """
    ax = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return kernel / kernel.sum()

def gaussianFilter1D(data, *, length, kind="valid", sigma=1):
    """
    Applies a 1D Gaussian filter to the input data. 

    [length] defines the size of the kernel and can be any size but, should be smaller than the input array.
    [kind] can be "valid", "full" or "same", by default it is set to "valid". [sigma] determines the
    standard deviation of the kernel.
    """
    x = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    kernel = np.exp(-0.5 * x**2 / sigma**2)
    return np.convolve(data, kernel / kernel.sum(), mode=kind)

def gaussianFilter2D(data, *, windowShape, kind="valid"):
    """
    Applies a 2D Gaussian filter to the input data.
    
    [windowShape] has to be a symetrical shape. -> the shape should be MxM.
    [kind] can be "valid", "full" or "same", by default it is set to "valid".
    """
    if kind == "full":
        paddingSize = windowShape[0] - 1
        data = np.pad(data, pad_width=(paddingSize, paddingSize))
    if kind == "same":
        paddingSize = (windowShape[0] - 1) // 2
        if windowShape[0] % 2 != 0:
            data = np.pad(data, pad_width=((paddingSize, paddingSize), (paddingSize, paddingSize)))
        else:
            data = np.pad(data, pad_width=((paddingSize, paddingSize + 1), (paddingSize, paddingSize + 1)))
    newWindow = np.lib.stride_tricks.sliding_window_view(data, window_shape=windowShape)
    return (_normalizingImages((newWindow * _gaussianKernel(windowShape[0])).sum(axis=(2, 3))) * 255).astype(np.uint8)

def gaussianFilter3D(data, *, windowShape, kind="valid"):
    """
    Applies a 3D Gaussian filter to the input data.
    
    [windowShape] has to be symetrical -> MxM.
    [kind] can be "valid", "full", or "same", by default it is set to "valid".
    """
    extractedData, temp = _extractingColourChannels(data), []
    for colorChannel in extractedData:
        temp.append(gaussianFilter2D(colorChannel, windowShape=windowShape, kind=kind))
    return np.array(temp).transpose((1, 2, 0))

def _extractingColourChannels(image):
    """
    Extracts the three color channels from an input image.
    Returns a tuple of three 2D arrays. Each array represents one color channel (Red, Green, Blue)
    of the input image.
    """
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]

def _normalizingImages(image):
    """
    Normalizes the pixel intensities in an image to the range [0, 1].
    """
    return (image - image.min()) / image.max()

def extractCoordinates(image):
    """
    Extracts the coordinates for every pixel. Returns y-cords, x-cords in a 2d array.
    """
    height, width = image.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return np.hstack((y.reshape(-1, 1), x.reshape(-1,1)))

def extractRGB(image, cords):
    """
    Takes an RGB-Image and a 2d array of cords. Extracts for every row in cords the corresponding RGB-Value-pair.
    Returns a 2d array of RGB-Values.
    """
    y, x = cords[:, 0], cords[:, 1]
    return image[y, x]