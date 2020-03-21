import os
import sys


from os import listdir
from os.path import isfile, join
from matplotlib.pyplot import imread
from multiprocessing import pool
 
from multiprocessing.dummy import Pool as ThreadPool


def get_celebA():
	BASE_PATH = "/nitthilan/data/celebA_hq_dattaset/download-celebA-HQ/celebA/Img/img_celeba/"
	onlyfiles = [join(BASE_PATH, f) for f in listdir(BASE_PATH) if isfile(join(BASE_PATH, f))]
	return onlyfiles

def get_file_shape(file_list):
	for file in file_list[:10]:
		img = imread(file)
		print(file, img.shape)
	return
print(len(get_celebA()))


#!/usr/bin/python
from PIL import Image
import os, sys
# Create thread pool
 

src_path = "/nitthilan/data/celebA_hq_dattaset/download-celebA-HQ/celebA/Img/img_celeba/"	
resize_path = "/nitthilan/data/celebA_hq_dattaset/download-celebA-HQ/celebA/Img/img_celeba_256px/"
resize_px = 256

def resize(item):
	if os.path.isfile(src_path+item):
		im = Image.open(src_path+item)
		f, e = os.path.splitext(item)
		# print(e)
		imResize = im.resize((resize_px,resize_px), Image.ANTIALIAS)
		imResize.save(resize_path+f + '_resized.jpg', 'JPEG', quality=95)


# resize(os.listdir(src_path), src_path, resize_path, resize_px)
# pool = ThreadPool(12)
# pool.map( resize, os.listdir(src_path) )

# get_file_shape(get_celebA())




import os
import sys


from os import listdir
from os.path import isfile, join
from matplotlib.pyplot import imread
BASE_PATH_MOBILE = "/nitthilan/data/clic/low_bitrate/mobile/train"
BASE_PATH_PFRAME = "/nitthilan/data/clic/p_frame_compress/data_p_frame/" #910, 1440/3=480, 1280x720
folder_list = ["Animation_1080P-6a33", "Animation_720P-0116", "HDR_2160P-06ae"]
# "low_bitrate", "p_frame_compress/data_p_frame/"

# idx = 2
# onlyfiles = [f for f in listdir(os.path.join(BASE_PATH_PFRAME, folder_list[idx])) if isfile(join(BASE_PATH_PFRAME, folder_list[idx], f))]

# for file in onlyfiles:
# 	img = imread(os.path.join(BASE_PATH_PFRAME, folder_list[idx], file))
# 	print(file, img.shape)



def get_path(is_mobile, is_train):
	BASE_PATH = "/nitthilan/data/clic/low_bitrate/"
	if(is_train):
		folder1 = "train"
	else:
		folder1 = "valid"

	if(is_mobile):
		folder2 = "mobile"
	else:
		folder2 = "professional"

	base_folder_path = join(BASE_PATH, folder2, folder1)
	onlyfiles = [join(base_folder_path, f) for f in listdir(base_folder_path) if isfile(join(base_folder_path, f))]

	return onlyfiles

def get_file_shape(file_list):
	for file in file_list:
		img = imread(file)
		print(file, img.shape)
	return


# 41 585 61 1048
print(len(get_path(0, 0)))
print(len(get_path(0, 1)))
print(len(get_path(1, 0)))
print(len(get_path(1, 1)))

get_file_shape(get_path(1,1))
