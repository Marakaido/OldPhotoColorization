import glob
import sys
import cv2
import numpy as np
from filtering import transforms

folder = sys.argv[1]
np.random.seed(0)

for t in ['train', 'test', 'val']:
    print(f'Processing folder {t}...')
    images = sorted(glob.glob(f'{folder}/{t}/*.jpg'))
    for i, image in enumerate(images):
        name = f'{i}.jpg'
        print(f'{image} --> {name}')
        img = cv2.imread(image)
        img = cv2.resize(img, (256,256))
        
        # Save original (ground truth) image
        cv2.imwrite(f'B/{t}/{name}', img)
        
        # Transform image
        img = transforms.process_img(img)

        # Save the transformed image
        cv2.imwrite(f'A/{t}/{name}', img)
