import glob
import random
import sys
import PIL
from PIL import Image

folder = sys.argv[1]

for t in ['train', 'test', 'val']:
    print(f'Processing folder {t}...')
    images = glob.glob(f'{folder}/{t}/*.jpg')
    for i, image in enumerate(images):
        name = f'{i}.jpg'
        with open(image, 'rb') as file:
            print(f'{image} --> {name}')
            img = Image.open(file).resize((256,256), resample=PIL.Image.LANCZOS)
            # Save original (ground truth) image
            img.save(f'A/{t}/{name}')
            # Save the transformed image
            img.save(f'B/{t}/{name}')
