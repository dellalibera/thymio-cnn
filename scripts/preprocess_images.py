from PIL import Image, ImageOps
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Images')
    parser.add_argument('--src', help='Source Images Directory', required=True)
    parser.add_argument('--output', help='Output Images Directory', default="./")

    args = parser.parse_args()
    src = args.src
    output = args.output

    size = 128, 96

    for root, dirs, files in os.walk(src):
        for file in files:
            im = Image.open(os.path.join(root, file))
            resize_im = ImageOps.fit(im, size, Image.ANTIALIAS)
            resize_im.format = im.format
            resize_im.save('{}/{}'.format(output, file))
    print("Finished!")