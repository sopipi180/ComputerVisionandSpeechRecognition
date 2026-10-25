import numpy as np
from scipy import ndimage

LUMINOSITY = np.array([0.2126, 0.7152, 0.0722])


def to_gray(img):
    """RGB (or already-gray) image -> 2-D float array in [0, 1]."""
    a = np.asarray(img, dtype=np.float64)
    if a.max() > 1.0:
        a = a / 255.0
    return a @ LUMINOSITY if a.ndim == 3 else a


def stretch(a):
    """Rescale to [0, 1] just for display - never feed this to a model."""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

KERNELS = {
    "blur (box 5x5)": np.ones((5, 5)) / 25,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "laplacian (all edges)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "sobel x (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "sobel y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
}


def apply_kernel(gray, kernel):
    """Convolve one kernel over the image. Edges are handled by mirroring."""
    return ndimage.convolve(gray, kernel, mode="reflect")


def filter_bank(gray, kernels=None):
    """{filter name: filtered image} for every kernel in KERNELS."""
    kernels = KERNELS if kernels is None else kernels
    return {name: apply_kernel(gray, k) for name, k in kernels.items()}


# Gaussian blur
def gaussian(gray, sigma, truncate=4.0):
    """scipy's Gaussian blur. sigma = the width of the bell curve, in pixels."""
    return ndimage.gaussian_filter(gray, sigma=sigma, truncate=truncate)


def gaussian_window(sigma, truncate=4.0):
    """How wide the kernel scipy actually builds is: 2*ceil(truncate*sigma)+1."""
    return 2 * int(truncate * float(sigma) + 0.5) + 1


# Sobel


def sobel(gray, sigma=0.0):
    g = gaussian(gray, sigma) if sigma else gray
    gy = ndimage.sobel(g, axis=0)   # change going down
    gx = ndimage.sobel(g, axis=1)   # change going right
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def sobel_edges(gray, sigma=0.0, threshold=None):
    """Sobel as a yes/no edge map, so it can be compared with Canny."""
    mag, _ = sobel(gray, sigma)
    t = threshold if threshold is not None else 0.2 * mag.max()
    return mag >= t


# Canny
# Canny's five steps, written out rather than called from a library:
#   1. blur, so noise is not detected as edges
#   2. gradients (Sobel)
#   3. non-maximum suppression  -> thin the thick ridges to 1 pixel
#   4. two thresholds           -> "strong" and "weak" edge pixels
#   5. hysteresis               -> keep weak pixels only if joined to a strong one

def _shift(a, dy, dx):
    """`a` moved by (dy, dx) pixels, with zeros arriving at the border."""
    h, w = a.shape
    out = np.zeros_like(a)
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def non_maximum_suppression(mag, angle):
    deg = np.rad2deg(angle) % 180   
    neighbour = {0: (0, 1), 45: (-1, 1), 90: (1, 0), 135: (1, 1)}
    binned = np.select([(deg < 22.5) | (deg >= 157.5), deg < 67.5, deg < 112.5],
                       [0, 45, 90], default=135)

    keep = np.zeros(mag.shape, bool)
    for direction, (dy, dx) in neighbour.items():
        here = binned == direction
        ahead = _shift(mag, -dy, -dx)
        behind = _shift(mag, dy, dx)
        keep |= here & (mag >= ahead) & (mag >= behind)
    return np.where(keep, mag, 0.0)


def hysteresis(mag, low, high):
    strong = mag >= high
    candidates = mag >= low
    blobs, _ = ndimage.label(candidates, structure=np.ones((3, 3)))
    keep = np.unique(blobs[strong])
    return np.isin(blobs, keep[keep > 0])


def canny(gray, sigma=1.5, low_ratio=0.10, high_ratio=0.20, thresholds=None):
    mag, angle = sobel(gray, sigma=sigma)          # steps 1-2
    thin = non_maximum_suppression(mag, angle)     # step 3

    if thresholds is not None:
        low, high = thresholds
    else:
        if not 0 <= low_ratio <= high_ratio <= 1:
            raise ValueError("need 0 <= low_ratio <= high_ratio <= 1")
        peak = thin.max()
        if peak == 0:
            return np.zeros(gray.shape, bool)
        low, high = low_ratio * peak, high_ratio * peak

    return hysteresis(thin, low, high)             # steps 4-5


def canny_peak(gray, sigma=1.5):
    """The largest thinned gradient in an image - the scale `canny` thresholds against."""
    return float(non_maximum_suppression(*sobel(gray, sigma=sigma)).max())


# numbers for comparing
def edge_energy(gray, sigma=1.0):
    """Mean gradient magnitude: how much edge there is in the image overall."""
    mag, _ = sobel(gray, sigma=sigma)
    return float(mag.mean())


def texture_score(gray, sigma=1.0):
    return float(ndimage.gaussian_laplace(gray, sigma=sigma).var())


def edge_density(edges):
    """Fraction of pixels an edge detector marked."""
    return float(np.asarray(edges, bool).mean())