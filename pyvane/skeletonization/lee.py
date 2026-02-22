from skimage.morphology import skeletonize as skel

# TODO: Add soft skeletonization on GPU

def skeletonize(img):
    """Skeletonizes a binary image using Lee's algorithm."""
    return skel(img).astype(img.dtype)

