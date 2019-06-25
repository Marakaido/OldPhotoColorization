import glob
import random
import sys
import PIL
from PIL import Image

color_mods = ['sepia', 'vignette', 'monochrome']
overlay_mods = ['dust', 'streaks']

folder = sys.argv[1]

for t in ["train", "test", "val"]:
        print(f'Processing folder {t}:')
        images = glob.glob(f'{folder}/{t}/*.jpg')
        for i, image in enumerate(images):
                with open(image, 'rb') as file:
                        print(f'{image} -> {i}.jpg')
                        img = Image.open(file).resize((256,256), resample=PIL.Image.LANCZOS)
                        # Save original image
                        img.save(f'A/{t}/{i}.jpg')

                        # apply color transformation
                        # apply overlay transformation

                        # save the transformed image
                        img.save(f'B/{t}/{i}.jpg')

        