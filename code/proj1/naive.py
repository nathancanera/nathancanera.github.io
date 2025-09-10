# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import img_as_ubyte
from logic import *

def jpg(imname): 
    # name of the input file
    impath = f'code/proj1/data/{imname}.jpg'

    # read in the image
    im = skio.imread(impath)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    def crop_image(img, top=0.0075, bottom=0.0075, left=0.015, right=0.015):
        H, W = img.shape
        top_px = int(H * top)
        bottom_px = int(H * bottom)
        left_px = int(W * left)
        right_px = int(W * right)

        return img[top_px:H-bottom_px, left_px:W-right_px]
    im = crop_image(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    b = crop_image(b, top=0.0025, bottom=0.0025, left=0.015, right=0.015)
    g = crop_image(g, top=0.0025, bottom=0.0025, left=0.015, right=0.015)
    r = crop_image(r, top=0.0025, bottom=0.0025, left=0.015, right=0.015)
    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    aligned_G, shift_G = align(g, b, metric=ncc)
    aligned_R, shift_R = align(r, b, metric=ncc)

    # create a color image
    im_out = np.dstack([aligned_R, aligned_G, b])
    im_out_norm = (im_out - im_out.min()) / (im_out.max() - im_out.min())
    im_out_uint8 = img_as_ubyte(im_out_norm)
    # save the image
    fname = f'code/proj1/out_path/{imname}.jpg'
    skio.imsave(fname, im_out_uint8)

    # display the image
    # skio.imshow(im_out_uint8)
    skio.show()
