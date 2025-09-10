import numpy as np
import cv2

def crop_image(img, top=0.0075, bottom=0.0075, left=0.015, right=0.015):
    H, W = img.shape
    top_px = int(H * top)
    bottom_px = int(H * bottom)
    left_px = int(W * left)
    right_px = int(W * right)

    return img[top_px:H-bottom_px, left_px:W-right_px]

def ssd(img1, img2):
    diff = img1 - img2
    return np.sqrt(np.sum(diff ** 2)) 

def ncc(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    numerator = np.sum(img1_flat * img2_flat)
    denominator = np.sqrt(np.sum(img1_flat ** 2) * np.sum(img2_flat ** 2))
    return numerator / denominator if denominator != 0 else 0


def align(match, base, window=15, metric=ssd):
    best_score = None
    best_shift = (0, 0)
    best_aligned = match
    
    for dx in range(-window, window+1):
        for dy in range(-window, window+1):
            shifted = np.roll(np.roll(match, dx, axis=0), dy, axis=1)
            
            h, w = shifted.shape
            border = int(0.1 * min(h, w))
            ref_crop = base[border:-border, border:-border]
            shifted_crop = shifted[border:-border, border:-border]

            score = metric(ref_crop, shifted_crop)
            
            if metric == ssd:
                better = (best_score is None) or (score < best_score)
            else:
                better = (best_score is None) or (score > best_score)
            
            if better:
                best_score = score
                best_shift = (dx, dy)
                best_aligned = shifted
    
    return best_aligned, best_shift

def pyramid(match, base, max_levels=5, window=15, metric=ssd):
    h, w = base.shape

    if min(h, w) < 300 or max_levels == 0:
        aligned, shift = align(match, base, window=window, metric=metric)
        return aligned, shift

    match_small = cv2.resize(match, (w // 2, h // 2), interpolation=cv2.INTER_AREA) #maybe do inter_nearest if slow
    base_small = cv2.resize(base, (w // 2, h // 2), interpolation=cv2.INTER_AREA) #maybe do inter_nearest if slow

    bruh, shift_small = pyramid(match_small, base_small,
                                max_levels=max_levels - 1,
                                window=window, metric=metric)

    dx, dy = shift_small[0] * 2, shift_small[1] * 2

    bruh2, refined_shift = align(match, base,
                                 window=2,
                                 metric=metric)

    dx += refined_shift[0]
    dy += refined_shift[1]

    final_aligned = np.roll(np.roll(match, dx, axis=0), dy, axis=1)

    return final_aligned, (dx, dy)