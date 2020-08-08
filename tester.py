import math
import os
import timeit

import cv2
import numpy as np

import opencvtest
import randomimage

dbfolder = "database"

imsize = (800, 600)

images = os.listdir(dbfolder)

debug = False


def get_image(enable_debug):
    img, filename, p, a = randomimage.random_image(enable_debug, dbfolder, imsize)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA), filename, p, a


def absdist(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


class TestRun:
    image: np.array = None
    filename: str = None
    angle: float = None
    pos: (float, float) = None
    estim_filename: str = None
    estim_angle: float = None
    estim_pos: (float, float) = None

    def __init__(self):
        self.image, self.filename, self.pos, self.angle = get_image(debug)

    def run_estim(self):
        self.estim_filename, self.estim_pos, self.estim_angle = opencvtest.try_all(debug, dbfolder, images, self.image)

    def check_estim(self):
        return self.filename == self.estim_filename, absdist(self.pos, self.estim_pos), math.fabs(
            self.angle - self.estim_angle)

    def __str__(self):
        return f"pos : {self.pos}, angle: {self.angle}, filename: {self.filename}, estim_pos: {self.estim_pos}, estim_filename: {self.estim_filename}, estim_angle: {self.estim_angle}"


test_array = []


def single_run():
    run = TestRun()
    run.run_estim()
    if len(test_array) % 10 == 0:
        print(len(test_array))
    test_array.append(run)


def all_runs():
    num = 50 if not debug else 1
    print(f"Starting {num} tests")
    time = timeit.timeit(single_run, number=num)
    print(f"Done, mean time is {time / num}")
    objp = 0
    meanposdiff = 0
    meananglediff = 0
    for test in test_array:
        o, p, a = test.check_estim()
        if o:
            objp += 1
        meanposdiff += p
        meananglediff += a
    objp = (objp / num) * 100
    meananglediff = meananglediff / num
    meanposdiff = meanposdiff / num
    print(f"Percentage of correct object {objp}")
    print(f"Mean angle diff {meananglediff}")
    print(f"Mean pos diff {meanposdiff}")


all_runs()
