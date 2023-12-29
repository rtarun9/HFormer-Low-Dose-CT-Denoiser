import numpy as np
import scipy.signal

def compute_MSE(img1, img2):
    return np.mean(np.square(img1 - img2))

def compute_PSNR(img1, img2, data_range):
    mse_ = compute_MSE(img1, img2)
    return 10 * np.log10((data_range ** 2) / mse_)

def create_window(window_size, channel):
    # Assuming you have a proper implementation of create_window for NumPy
    # You can use a Gaussian window as an example
    window = np.outer(signal.gaussian(window_size, std=1), np.ones(channel))
    return window / np.sum(window)

def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    if len(img1.shape) == 2:
        shape_ = img1.shape[-1]
        img1 = np.reshape(img1, (1, shape_, shape_, 1))
        img2 = np.reshape(img2, (1, shape_, shape_, 1))

    window = create_window(window_size, channel)

    mu1 = scipy.signal.convolve2d(img1, window, mode='same')
    mu2 = scipy.signal.convolve2d(img2, window, mode='same')

    mu1_sq, mu2_sq = np.square(mu1), np.square(mu2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = scipy.signal.convolve2d(np.square(img1), window, mode='same') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.square(img2), window, mode='same') - mu2_sq
    sigma12 = scipy.signal.convolve2d(img1 * img2, window, mode='same') - mu1_mu2

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return np.mean(ssim_map)
    else:
        return np.mean(np.mean(np.mean(ssim_map, axis=1), axis=1), axis=1)
