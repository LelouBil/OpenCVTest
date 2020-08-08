import os
import random

from PIL import Image


def _get_random_image(debug, folder):
    imgs = os.listdir(folder)
    random_image_index = random.randrange(0, len(imgs))
    chosen = Image.open(os.path.join(folder, imgs[random_image_index]))
    chosen = chosen.convert("RGBA")
    if debug:
        chosen.show()
    return chosen, imgs[random_image_index]


def _get_random_angle():
    return random.randrange(-180, 180)


def _rotate_image(image, angle):
    rotated = image.rotate(angle, expand=True)
    mask = Image.new("RGBA", rotated.size, (255,) * 4)
    rotated = Image.composite(rotated, mask, rotated)
    return rotated


def _get_random_coords(imsize, basesize):
    x, y = imsize
    bw, bh = basesize
    rndx = random.randrange(0, bw - x)
    rndy = random.randrange(0, bh - y)
    cx = round(x / 2) + rndx
    cy = round(y / 2) + rndy
    return (rndx, rndy), (cx, cy)


def random_image(debug, folder, basesize):
    baseimage = Image.new('RGBA', basesize, (255, 0, 255))

    chosen, filename = _get_random_image(debug, folder)
    angle = _get_random_angle()

    rotated = _rotate_image(chosen, angle)

    topleft, center = _get_random_coords(rotated.size, baseimage.size)
    baseimage.paste(rotated, topleft)
    if debug:
        baseimage.show()
    return baseimage, filename, center, angle
