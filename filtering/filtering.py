import cv2 as cv
import numpy as np
import os

VIGNETTE_PROB = 0.3
VIGNETTE_SIGMA = 200
OVERLAY_MAX_STRENGTH = 0.4
GRAY_PROB = 0.6
SEPIA_COL_FACTOR = 0.2
SEPIA_BRIGHT_FACTOR = 0.1

def grey(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

def sepia(img, color_factor, brightness_factor):
    # create basic sepia filter kernel
    sepia = np.array([[0.131,0.534,0.272]
            ,[0.168,0.686,0.349]
            ,[0.189,0.769,0.393]])

    # make some deviations possible
    # vary colors slightly
    sepia = sepia * (np.random.rand(3,3) * color_factor + (1 - color_factor/2))

    # vary brightness slightly
    sepia = sepia * (np.random.rand() * brightness_factor + (1 - brightness_factor/2))

    return cv.transform(img,sepia);

def vignette(img, sigma):
    vignette = cv.getGaussianKernel(img.shape[1],sigma)
    vignette = cv.getGaussianKernel(img.shape[0],sigma) * vignette.T
    vignette = 255 * vignette / np.linalg.norm(vignette)

    img_tmp = np.array(img,dtype=np.uint16)

    img_tmp[:,:,0] = img[:,:,0] * vignette
    img_tmp[:,:,1] = img[:,:,1] * vignette
    img_tmp[:,:,2] = img[:,:,2] * vignette

    img_tmp = 2*img_tmp + img
    img_tmp = np.array(img_tmp/3, dtype=np.uint8)

    return img_tmp

def overlay_artifacts(img, strength):
    # puts random dust image over image
    # todo: maybe add rotation back in
    # (but would require new cropping to get rid of black boundaries)
    # possible: mirroring of image

    # choose random dust image out of those provided in folder
    if os.getcwd()[-9:] != "filtering":
        dust_path = 'filtering/dust/' + np.random.choice(os.listdir('filtering/dust/'))
    else:
        dust_path = 'dust/' + np.random.choice(os.listdir('dust/'))
    # read out only alpha channel (3) so opacity will be understood as grayscale
    dust_img = cv.imread(dust_path, flags=-1)[:,:,3]

    # make less prominent by division by 2, change int range to avoid problems
    dust_img = np.array(dust_img*strength,dtype=np.uint16)

    dust_dims = dust_img.shape
    img_dims = img.shape

    # factor that denotes difference in image/dust image dimensions
    scaling_factor = min(dust_dims[0]/img_dims[0],dust_dims[1]/img_dims[1])

    # choose random patch, width maximally dust image width
    # or 1.5 original image width (otherwise, dust gets too small in rescaling)
    patch_size_x = np.random.randint(img_dims[0],min(1.5*img_dims[0],int(img_dims[0]*scaling_factor)))

    # choose patch height with fitting aspect ratio
    patch_size_y = int((patch_size_x/dust_dims[0]) * dust_dims[1])

    # select random patch location. no wrap-around for now
    loc_x = np.random.randint(0,dust_dims[0] - patch_size_x)
    loc_y = np.random.randint(0,dust_dims[1] - patch_size_y)

    # cut out patch
    patch = dust_img[loc_x:loc_x + patch_size_x, loc_y:loc_y + patch_size_y]

    # potential mirroring/rotation
    flipcode = np.random.randint(-1,3)
    if flipcode != 2:
        patch = cv.flip(patch,flipcode)

    # resize
    patch = cv.resize(patch,(img_dims[1],img_dims[0]))

    # get maximal rgb values from image
    # adjust patch s.t. 255 won't be exceeded adding the dust to every channel
    img_maxval = np.max(img,axis=2)
    patch = np.where(patch + img_maxval > 255,np.ones(img_dims[0:2],dtype = np.uint8) * 255 - img_maxval,patch)

    # add dust patch
    img[:,:,0] += patch
    img[:,:,1] += patch
    img[:,:,2] += patch

    return img

def process_img(img):
    if np.random.rand() < VIGNETTE_PROB:
        img = vignette(img,VIGNETTE_SIGMA)

    img = overlay_artifacts(img, np.random.rand() * OVERLAY_MAX_STRENGTH)

    if np.random.rand() < GRAY_PROB:
        img = grey(img)
    else:
        img = sepia(img, SEPIA_COL_FACTOR, SEPIA_BRIGHT_FACTOR)

    return img

def main():

    # stupid workaround to load actual existing images
    # some are corrupted but all of them start with . - maybe thumbnails or smth
    imgname = '.'
    while (imgname[0] == '.'):
        imgname = np.random.choice(os.listdir('train/'))

    img_path = 'train/' + imgname
    img = cv.imread(img_path)

    print("Loaded image:", img_path)
    cv.imshow('original',img)
    cv.waitKey()

    print("Image shape:", img.shape)

    img = process_img(img)

    #cv.imshow('filtered with opencv',img_f)
    cv.imshow('filtered + dust',img)

    cv.waitKey()

    cv.imwrite(str(imgname)[:-4]+'_filtered.jpg',img)
    print('Written to: '+str(imgname)[:-4]+'_filtered.jpg')

if __name__ == '__main__':
    main()


#def white_streaks(img):
