import numpy as np
from PIL import Image
from scipy import ndimage
 
IMAGE_SIZE = (200, 150)     # (width, height) - every MILK10k image is 600x450, so exactly 1/3
HIST_BINS = 8
GLCM_LEVELS = 16
CROP_FRACTION = 0.6         # the middle 60% of the frame
 
 
def load_rgb(path, size=IMAGE_SIZE, crop=None):
    """
    Read one image as a float RGB array, always the same size.
 
    crop=0.6 keeps the middle 60% of the frame before resizing. The lesion is
    roughly centred and fills only part of the photo, so without a crop these
    features partly describe healthy skin and the dermatoscope's background.
    Cropping first also means the lesion is resized from more pixels, not fewer.
    """
    with Image.open(path) as handle:
        handle.draft("RGB", (size[0] * 2, size[1] * 2))   # fast rough decode
        im = handle.convert("RGB")
    if crop:
        w, h = im.size
        cw, ch = int(w * crop), int(h * crop)
        left, top = (w - cw) // 2, (h - ch) // 2
        im = im.crop((left, top, left + cw, top + ch))
    return np.asarray(im.resize(size, Image.BILINEAR), dtype=np.float64)
 
 
def to_gray(rgb):
    """Same grayscale as the class notebook: a plain average of the channels."""
    return rgb.mean(axis=2)
 
 
#  colour
 
 
def color_stats(rgb):
    """6 numbers: how dark the lesion is, and how varied its colour is."""
    out = {}
    for i, c in enumerate("RGB"):
        channel = rgb[:, :, i]
        out[f"mean_{c}"] = channel.mean()
        out[f"std_{c}"] = channel.std()
    return out
 
 
def color_histogram(rgb, bins=HIST_BINS):
    """
    24 numbers: how many pixels fall in each brightness bin, per channel.
 
    Divided by the number of pixels, so two images of different size are
    comparable - exactly the count -> fraction step from the slides.
    """
    out = {}
    for i, c in enumerate("RGB"):
        counts, _ = np.histogram(rgb[:, :, i], bins=bins, range=(0, 256))
        for b, value in enumerate(counts / rgb[:, :, i].size):
            out[f"hist_{c}{b}"] = value
    return out
 
 
#  texture
 
 
def texture_features(rgb):
    """
    3 numbers, the class notebook's function.
 
    edge_density   mean Sobel magnitude     - how much border and fine detail
    roughness      mean local variance      - rough vs uniform
    laplacian_var  variance of the Laplacian - sharp detail vs blurry
    """
    gray = to_gray(rgb)
 
    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    edge_density = np.hypot(gx, gy).mean()
 
    # local variance in every 9x9 window at once: mean(x^2) - mean(x)^2
    local_mean = ndimage.uniform_filter(gray, size=9)
    local_sq_mean = ndimage.uniform_filter(gray ** 2, size=9)
    roughness = (local_sq_mean - local_mean ** 2).mean()
 
    laplacian_var = ndimage.laplace(gray).var()
    return {"edge_density": edge_density, "roughness": roughness,
            "laplacian_var": laplacian_var}
 
 
#  GLCM
 
 
def gray_cooccurrence(gray, levels=GLCM_LEVELS):
    """
    How often a grey level sits next to another grey level (Haralick, 1973).
 
    The image is first reduced to `levels` grey levels, otherwise the matrix
    would be 256x256 and almost empty. Four directions are counted together
    (right, down, and the two diagonals) and the result is made symmetric, so
    the features do not depend on how the photo happens to be rotated.
    """
    q = np.clip((gray / (256 / levels)).astype(int), 0, levels - 1)
    h, w = q.shape
    matrix = np.zeros((levels, levels), dtype=np.float64)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # the pixels that still have a neighbour at this offset, and those neighbours
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        a = q[y0:y1, x0:x1]
        b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        np.add.at(matrix, (a.ravel(), b.ravel()), 1)
    matrix += matrix.T                       # count each pair both ways
    total = matrix.sum()
    return matrix / total if total else matrix
 
 
def glcm_features(rgb, levels=GLCM_LEVELS):
    """5 numbers summarising that matrix."""
    p = gray_cooccurrence(to_gray(rgb), levels)
    i, j = np.mgrid[0:levels, 0:levels]
    diff = i - j
 
    contrast = float((p * diff ** 2).sum())           # big jumps between neighbours
    homogeneity = float((p / (1.0 + diff ** 2)).sum())  # neighbours look alike
    energy = float(np.sqrt((p ** 2).sum()))           # few pairs dominate = uniform
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())   # disorder
 
    # correlation: are a pixel and its neighbour linearly related?
    mu_i, mu_j = (p * i).sum(), (p * j).sum()
    sd_i = np.sqrt((p * (i - mu_i) ** 2).sum())
    sd_j = np.sqrt((p * (j - mu_j) ** 2).sum())
    correlation = float(((p * (i - mu_i) * (j - mu_j)).sum() / (sd_i * sd_j))
                        if sd_i > 0 and sd_j > 0 else 0.0)
 
    return {"glcm_contrast": contrast, "glcm_homogeneity": homogeneity,
            "glcm_energy": energy, "glcm_entropy": entropy,
            "glcm_correlation": correlation}
 
 
# shape
 
 
def asymmetry(rgb):
    """
    2 numbers, the "A" of the ABCDE rule.
 
    A benign mole tends to look the same on both sides; a melanoma often does
    not. Without a lesion mask (segmentation is Session 8) this is measured on
    the whole image: the distance between the average colour of the two halves.
    """
    h, w, _ = rgb.shape
    left, right = rgb[:, :w // 2], rgb[:, w - w // 2:]
    top, bottom = rgb[:h // 2], rgb[h - h // 2:]
    return {
        "asym_lr": float(np.linalg.norm(left.mean(axis=(0, 1)) - right.mean(axis=(0, 1)))),
        "asym_tb": float(np.linalg.norm(top.mean(axis=(0, 1)) - bottom.mean(axis=(0, 1)))),
    }
 
 
# all together
 
EXTRACTORS = {
    "color": color_stats,
    "hist": color_histogram,
    "texture": texture_features,
    "glcm": glcm_features,
    "shape": asymmetry,
}
 
 
def extract(path, crop=None):
    """One image -> one flat dict of every feature, with readable column names."""
    rgb = load_rgb(path, crop=crop)
    row = {}
    for fn in EXTRACTORS.values():
        row.update(fn(rgb))
    return row
 
 
def group_columns(columns):
    """{group name: [column names]} - which columns each feature group produced."""
    probe = np.zeros((4, 4, 3))
    groups = {}
    for name, fn in EXTRACTORS.items():
        wanted = set(fn(probe))
        groups[name] = [c for c in columns if c in wanted]
    return groups