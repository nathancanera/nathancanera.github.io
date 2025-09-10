# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import img_as_ubyte
from logic import *

def tif(imname):
    # name of the input file
    impath = f'data/{imname}.tif'

    # read in the image
    im = skio.imread(impath)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    im = crop_image(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    # b = crop_image(b, top=0.0025, bottom=0.0025, left=0.015, right=0.015)
    # g = crop_image(g, top=0.0025, bottom=0.0025, left=0.015, right=0.015)
    # r = crop_image(r, top=0.0025, bottom=0.0025, left=0.015, right=0.015)


    aligned_G, shift_G = pyramid(g, b, metric=ncc, max_levels=2, window=25)
    aligned_R, shift_R = pyramid(r, b, metric=ncc, max_levels=2, window=25)

    aligned_G, fine_shift_G = align(aligned_G, b, window=3, metric=ncc)
    aligned_R, fine_shift_R = align(aligned_R, b, window=3, metric=ncc)

    shift_G = (shift_G[0] + fine_shift_G[0], shift_G[1] + fine_shift_G[1])
    shift_R = (shift_R[0] + fine_shift_R[0], shift_R[1] + fine_shift_R[1])

    im_out = np.dstack([aligned_R, aligned_G, b])
    im_out_norm = (im_out - im_out.min()) / (im_out.max() - im_out.min())
    im_out_uint8 = img_as_ubyte(im_out_norm)

    fname = f'out_path/{imname}.jpg'
    skio.imsave(fname, im_out_uint8)

    skio.imshow(im_out_uint8)
    #skio.show()
