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

X_CAPT = np.float32([])
Y_CAPT = np.float32([])

downlevs_ = 2

ncircles_ = 20
nangles_ = 200
rstep_ = 12 # spacing (in pixels) between circles we sample
LAMBDA = 10 # estimated pixel noise
BETA = 100 # prior smoothness term
ALPHA = 0.99 # mean(t+1) = alpha * mean(t) + (1-alpha) frame(t)

# Global hacks
theta_end_ = 0
theta_start_ = 0

xmin_ = 0
xmax_ = 0
ymin_ = 0
ymax_ = 0

corner_x_ = None
corner_y_ = None

xq_ = None
yq_ = None
tq_ = None

amat_ = None

reg_ = None

dframe_ = None

meanpx_ = 0

gain_ = 0

backim_ = None

def setRegularizer():
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	xdim = amat_.shape[1]
	bmat = np.zeros((xdim-1, xdim), dtype=np.float64)

	for i in range(1, bmat.shape[0]):
		bmat[i, i] = 1
		bmat[i, i+1] = -1

	cmat = np.eye(xdim, xdim, dtype=np.float64)
	cmat[0, 0] = 0

	reg_ = bmat.T @ bmat + cmat

def setAmat():
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	nobs = tq_.shape[0] * tq_.shape[1]
	tdelta = (theta_end_ - theta_start_) / (nangles_ - 1)
	tdir = tdelta / np.abs(tdelta)

	thetas = np.zeros((1, nangles_), dtype=np.float32)
	tcur = theta_start_
	for i in range(nangles_):
		thetas[0, i] = tcur
		tcur += tdelta

	amat_ = np.zeros((nobs, nangles_ + 1), dtype=np.float64)
	for i in range(nobs):
		angle = tq_[i, 0]
		mask = ((angle - thetas) * tdir >= 0)
		# find the idx of the last theta we can see at this angle
		idx = int(np.sum(mask) - 1)
		if (idx < 0): # doesn't see any of the scene
			amat_[i, 0] = 1; # still sees the constant light
		else: # angle indexing starts at 1
			diff = (angle - thetas[0, idx]) / tdelta
			for j in range(idx + 1):
				amat_[i, j] = 1
			amat_[i, idx+1] = 0.5 * (2 - diff) * diff + 0.5
			if (idx < nangles_ - 1):
				amat_[i, idx+2] = 0.5 * diff * diff

def setPieSliceXYLocs():
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	xq = np.zeros((ncircles_, nangles_))
	yq = np.zeros((ncircles_, nangles_))
	tq = np.zeros((ncircles_, nangles_))
	 
	rcur = 0
	acur = 0
	astep = (theta_end_ - theta_start_) / (nangles_ - 1)

	for ci in range(ncircles_):
		rcur += rstep_
		acur = theta_start_

		for ai in range(nangles_):
			xq[ci, ai] = corner_x_ + rcur * np.cos(acur)
			yq[ci, ai] = corner_y_ + rcur * np.sin(acur)
			tq[ci, ai] = acur
			acur += astep

	xq_ = xq.reshape((ncircles_ * nangles_, 1)) 
	yq_ = yq.reshape((ncircles_ * nangles_, 1))
	tq_ = tq.reshape((ncircles_ * nangles_, 1))

def setObsXYLocs(nrows_, ncols_):
	global theta_end_
	global theta_start_
	global xmin_
	global xmax_
	global ymin_
	global ymax_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	setPieSliceXYLocs()

	xmin, xmax, _, _ = cv2.minMaxLoc(xq_)
	ymin, ymax, _, _ = cv2.minMaxLoc(yq_)

	pad = 5 * (1 << (downlevs_ + 1))
	xmin_ = np.fmax(0, round(xmin) - pad)
	ymin_ = np.fmax(0, round(ymin) - pad)
	xmax_ = np.fmin(ncols_, round(xmax) + pad)
	ymax_ = np.fmin(nrows_, round(ymax) + pad)

	xq_ = (xq_ - xmin_)/(1<<downlevs_)
	yq_ = (yq_ - ymin_)/(1<<downlevs_)

def setObsRegion(corner, wallpt, endpt):
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	print(corner)
	print(wallpt)
	print(endpt)

	corner_x_ = corner[0]
	corner_y_ = corner[1]

	wall_x = wallpt[0]
	wall_y = wallpt[1]

	end_x = endpt[0]
	end_y = endpt[1]

	wall_angle = np.arctan2(wall_y - corner_y_, wall_x - corner_x_)
	end_angle = np.arctan2(end_y - corner_y_, end_x - corner_x_)
	wall_angle = np.fmod(wall_angle + 2 * np.pi, 2 * np.pi)
	end_angle = np.fmod(end_angle + 2 * np.pi, 2 * np.pi)

	# wall_angle = np.degrees(wall_angle)
	# end_angle = np.degrees(end_angle)

	diff_angle = end_angle - wall_angle
	angle_dir = diff_angle / abs(diff_angle)
	print(f'selected point on wall at angle {wall_angle}')
	print(f'selected ending point at angle {end_angle}')

	theta_start_ = wall_angle
	if np.abs(diff_angle) < np.pi:
		print("keep the current dir from wall to end point\n")
		theta_end_ = wall_angle + diff_angle
	else:
		print("reverse the dir from wall to end point")
		diff_angle = 2 * np.pi - abs(diff_angle)
		theta_end_ = wall_angle - angle_dir * diff_angle

	print(f'starting angle: {theta_start_}, ending angle: {theta_end_}')

def findObsRegion(frame_):
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_
	
	if theta_start_ != theta_end_ and \
		corner_x_ != 0 and \
		corner_y_ != 0:
		
		print(f'Using input values: corner ({corner_x_}, {corner_y_}), thetas {theta_start_} to {theta_end_}')
		return

	if np.shape(frame_)[0] <= 0 or np.shape(frame_)[1] <= 0:
		print("ERROR: No frame has been read yet")

	print("Getting observation region from user")
	def capture_click(event, x_click, y_click, flags, params):
		global X_CAPT, Y_CAPT
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)
		if event == cv2.EVENT_LBUTTONDOWN:
			xy_click = np.float32([x_click, y_click])
			xy_click = xy_click.reshape(-1, 1, 2)
			#print(xy_click)
			I = cv2.cvtColor(frame_, cv2.COLOR_RGB2GRAY)
			refined_xy = cv2.cornerSubPix(I, xy_click, (11, 11), (-1, -1), criteria)
			#print(refined_xy)
			X_CAPT = np.append(X_CAPT, refined_xy[0, 0, 0])
			Y_CAPT = np.append(Y_CAPT, refined_xy[0, 0, 1])
			cv2.drawMarker(frame_, (int(X_CAPT[-1]), int(Y_CAPT[-1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30)

	compute_name = 'Calibration'
	cv2.namedWindow(compute_name)
	cv2.setMouseCallback(compute_name, capture_click)

	print("Please select corner, a point at the base of the wall, then an angular endpoint")
	while True:
		cv2.imshow(compute_name, frame_)
		key = cv2.waitKey(1)

		if key == ord("q") or len(X_CAPT) == 3:
			break

	print(X_CAPT, Y_CAPT)
	corner = np.asarray([X_CAPT[0], Y_CAPT[0]])
	wallpt = np.asarray([X_CAPT[1], Y_CAPT[1]])
	endpt = np.asarray([X_CAPT[2], Y_CAPT[2]])

	setObsRegion(corner, wallpt, endpt)

def setup(nrows_, ncols_):
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	setObsXYLocs(nrows_, ncols_)
	setAmat()
	setRegularizer()

	tmp = amat_.T @ (amat_ / LAMBDA) + BETA * reg_
	gain_ = np.linalg.inv(tmp) @ (amat_.T / LAMBDA)
	io.imsave("data/gain.png", (gain_ - np.min(gain_)) / (np.max(gain_) - np.min(gain_)))

def preprocessFrame(frame_):
	global theta_end_
	global theta_start_
	global xmin_
	global xmax_
	global ymin_
	global ymax_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	dframe_ = frame_
	dframe_ = dframe_[ymin_:ymax_, xmin_:xmax_]
	for i in range(downlevs_):
		dframe_ = cv2.pyrDown(dframe_)

def updateMeanImage(total):
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	if total == 1:
		backim_ = dframe_
	else:
		backim_ = ALPHA * backim_ + (1 - ALPHA) * dframe_

	meanpx_ = np.zeros(3)
	meanpx_[0] = np.mean(backim_[:, :, 0])
	meanpx_[1] = np.mean(backim_[:, :, 1])
	meanpx_[2] = np.mean(backim_[:, :, 2])

def plotObsXYLocs(name):
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global backim_
	global gain_

	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	sample = np.copy(dframe_)
	sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
	cv2.imshow(name, sample)

	cv2.circle(sample, (int((corner_x_ - xmin_) / (1 << downlevs_)), int((corner_y_ - ymin_) / (1 << downlevs_))), 1, (255,0,0))
	for i in range(ncircles_*nangles_):
		cv2.circle(sample, (round(xq_[i][0]), round(yq_[i][0])), 1, (255,0,0))

	cv2.imshow(name, sample)

def processFrame():
	global theta_end_
	global theta_start_
	global xmin_
	global ymin_
	global corner_x_
	global corner_y_
	global xq_
	global yq_
	global tq_
	global amat_
	global reg_
	global dframe_
	global meanpx_
	global gain_
	global backim_

	samples = cv2.remap(dframe_ - backim_, np.asarray(xq_, dtype=np.float32), np.asarray(yq_, dtype=np.float32), cv2.INTER_LINEAR)
	# samples = cv2.remap(I, xq_, yq_, cv2.INTER_LINEAR)
	
	dispchan = np.zeros((1, nangles_, 3), dtype=np.float64)
	for i in range(3):
		cur_out = (gain_ @ (samples[:, :, i] + meanpx_[i])).T
		dispchan[:, :, i] = cur_out[:, 1:amat_.shape[1]]

	return dispchan

