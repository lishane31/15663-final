import numpy as np
import cv2

DEFAULT_ROWS = 300

name_ = None
roll_up_ = False
nrows_ = DEFAULT_ROWS
ncols_ = 0

bufsize_ = 0
buffer_ = None

dstart_ = 0
drow_ = 0

def RollingDisplay(name, roll_up, ncols, buf_factor=2):
	global name_
	global roll_up_
	global nrows_
	global ncols_
	global bufsize_
	global buffer_ 
	global dstart_
	global drow_

	if buf_factor < 1:
		print("buffer must be at least same length as display")

	name_ = name;
	roll_up_ = roll_up;
	nrows_ = DEFAULT_ROWS;
	ncols_ = ncols;

	bufsize_ = nrows_ * buf_factor;
	buffer_ = np.zeros((bufsize_, ncols_), dtype=np.float64)
	cv2.namedWindow(name_, cv2.WINDOW_NORMAL)

	dstart_ = 0
	drow_ = 0

def nextrow():
	global name_
	global roll_up_
	global nrows_
	global ncols_
	global bufsize_
	global buffer_ 
	global dstart_
	global drow_

	if (drow_ >= bufsize_):
		wrapAround()

	dstart_ = np.maximum(0, drow_ - nrows_ + 1)
	drow_ += 1
	return buffer_[drow_ - 1]


def update(disp_):
	global name_
	global roll_up_
	global nrows_
	global ncols_
	global bufsize_
	global buffer_ 
	global dstart_
	global drow_

	if (roll_up_):
		cv2.flip(buffer_[dstart_:dstart_ + nrows_], disp_, 0)
	else:
		disp_ = buffer_[dstart_:dstart_ + nrows_]
	
	print(disp_)
	cv2.imshow(name_, disp_)


def wrapAround():
	global name_
	global roll_up_
	global nrows_
	global ncols_
	global bufsize_
	global buffer_ 
	global dstart_
	global drow_
	
	end = drow_
	start = end - (nrows_ - 1)
	last = buffer_[start:end]
	buffer_[0:nrows_ - 1] = last
	drow_ = nrows_ - 1
