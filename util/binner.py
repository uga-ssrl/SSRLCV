import cv2
import os
import sys

img_dir = '/media/firesauce/JAX_1TB/datasets/need_binning/ship_airbus_1.5m/test_v2/'
bin_dir = '/media/firesauce/JAX_1TB/datasets/need_binning/ship_airbus_1.5m/binned/'

def bin_in_folder():
	for filename in os.listdir(img_dir):
		img = cv2.imread(os.path.join(img_dir,filename))
		cv2.resize(img,(192,192))
		cv2.imwrite(os.path.join(bin_dir,filename),img)


if __name__ == '__main__':
	bin_in_folder()
		

