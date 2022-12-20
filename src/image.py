from asyncio import Task
import math
from pickletools import uint8
from skimage import io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
from scipy import interpolate
from scipy.signal import convolve2d
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource
import inference
import rolling_display as rd

nrows_ = 0
ncols_ = 0
frame_ = None
cap_ = None

def skipFrames(n):
	global cap_

	for i in range(n):
		ok, _ = cap_.read()
		
		if not ok:
			print("WARNING: could not skip frames")

def getNextFrame():
	global frame_
	global cap_

	ok, frame = cap_.read()
	if not ok:
		print("WARNING: could not grab frame")
	else:
		frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



def processVideo(
	vidname,
	disp_rollup, start_time,
	process_fps, end_time
):
	global nrows_
	global ncols_
	global frame_
	global cap_

	cap_ = cv2.VideoCapture(vidname)

	if not cap_.isOpened():
		print("ERROR: coult not open file")
		return

	vid_fps = round(cap_.get(5))
	start_f = start_time * vid_fps
	step = round(vid_fps / process_fps)
	end_f = round(end_time * vid_fps)
	end_f = np.minimum(end_f, int(cap_.get(7)))

	print(f'start {start_f}, step {step}, end {end_f}')

	skipFrames(start_f - step - 1)
	getNextFrame()

	nrows_ = np.shape(frame_)[0]
	ncols_ = np.shape(frame_)[1]
	print(f'{nrows_} x {ncols_} framesize', nrows_, ncols_)

	inference.findObsRegion(frame_)

	inference.setup(nrows_, ncols_)

	inference.preprocessFrame(frame_)
	inference.updateMeanImage(1)

	inference.plotObsXYLocs("sample locations")

	rd.RollingDisplay("output", disp_rollup, inference.nangles_)

	cv2.namedWindow("input", cv2.WINDOW_NORMAL)
	cv2.namedWindow("background image", cv2.WINDOW_NORMAL)

	nframes = (end_f - start_f)/step + 1

	print("Starting inference, press esc to break")

	outputImage = np.zeros((int(nframes), 200, 3))
	for i in tqdm(range(int(nframes))):
		skipFrames(step - 1)

		getNextFrame()

		if frame_.shape[0] == 0 or frame_.shape[1] == 0:
			break

		inference.preprocessFrame(frame_)

		cur_disp = rd.nextrow()
		cur_disp = inference.processFrame()

		# cur_disp = (cur_disp + 0.05) / 0.1;

		cv2.imshow("input", inference.dframe_ / 255)
		cv2.imshow("background image", inference.backim_ / 255)

		rd.update(cur_disp)
		outputImage[i, :, :] = cur_disp[0, :, :]

		if (cv2.waitKey(1) == 27):
			print("pressed ESC key, exiting inference")
			break

		inference.updateMeanImage(i + 2)

	outputImage = (outputImage - np.min(outputImage)) / (np.max(outputImage) - np.min(outputImage))
	io.imsave(f'data/stuff.png', outputImage)
	cv2.destroyAllWindows()


if __name__ == '__main__':
    processVideo("indoors/loc1_one_person_walking.MOV", False, 2, 20, 26)
	