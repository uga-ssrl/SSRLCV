import cv2
import os
import sys
import argparse

min_gsd = 7
max_gsd = 20

interpolation_algos = [cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]

def scale_dataset(folder,destination,scaling_factor,algo):
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		width,height,channels = img.shape
		resized = cv2.resize(img,(int(width*scaling_factor + 1),int(height*scaling_factor +1)),algo)
		cv2.imwrite(os.path.join(destination,filename),resized)

def get_args():
	parser = argparse.ArgumentParser(description='Prep Satellite Imagery Datasets for object detection',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', '--input', metavar='IN', type=str, required=True, help='input path', dest='input')
	parser.add_argument('-d', '--destination', metavar='D', type=str, required=True,help='destination path', dest='dest')
	parser.add_argument('-s', '--scale', metavar='S', type=float, required=True, help='ratio of input size to output size', dest='scale')
	parser.add_argument('-a', '--algorithm', metavar='A', type=int, default=1, help='interpolation algorith for opencv.resize()', dest='algo') 
	return parser.parse_args()



if __name__ == '__main__':
	args = get_args()
	assert args.scale > 0.0, 'scaling factor must be greater than 0'
	assert args.algo < 5, 'algo must be below 5'   
	scale_dataset(args.input,args.dest,args.scale,interpolation_algos[args.algo])
	


