import cv2
import os
import sys
import argparse
from glob import glob
import random
from shutil import copyfile


min_gsd = 7
max_gsd = 20

interpolation_algos = [cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4]
class_dict_count = {
	'mountain':0,
	'river':0,
	'wetland':0,
	'seaorlake':0,
	'island':0,
	'forest':0,
	'snoworice':0,
	'beach':0,
	'desert':0,
	'agricultural':0,
	'roadway':0,
	'bridge':0,
	'industrial':0,
	'commercial':0,
	'residential':0,
	'stadium':0,
	'golf_course':0,
	'baseball-diamond':0,
	'ground-track-field':0,
	'soccer-ball-field':0,
	'thermal-power':0,
	'plane':0,
	'ship':0,
	'harbor':0,
	'storage-tank':0,
	'small-vehicle':0,
	'large-vehicle':0,

}
class_dict = {
	'mountain':'mountain',
	'river':'river',
	'wetland':'wetland',
	'seaorlake':'seaorlake',
	'island':'island',
	'forest':'forest',
	'snoworice':'snoworice',
	'beach':'beach',
	'desert':'desert',
	'agricultural':'agricultural',
	'roadway':'roadway',
	'bridge':0,
	'industrial':'industrial',
	'commercial':'commercial',
	'residential':'residential',
	'stadium':'stadium',
	'golf_course':'golf_course',
	'baseball-diamond':1,
	'ground-track-field':2,
	'soccer-ball-field':2,
	'thermal-power':'thermal_power',
	'plane':3,
	'ship':5,
	'harbor':4,
	'storage-tank':6,
	'small-vehicle':7,
	'large-vehicle':7
}

def scaleDOTALabel(folder,label_dir,destination):
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		height,width,channels = img.shape
		id = os.path.splitext(filename)[0] + '.txt'
		label = os.path.join(label_dir,id)
		print('fixing ' + label)
		lines = open(label).readlines()
		i = 0
		out = open(os.path.join(destination,id),'w')
		for line in lines:
			values = line.split()
			width_box = (float(values[2]) - float(values[0]))/width
			height_box = (float(values[5]) - float(values[1]))/height
			center_box = [(float(values[0])/width) + (width_box/2),(float(values[1])/height) + (height_box/2)]
			if values[8] not in class_dict:
				continue
			cnn_class = class_dict[values[8]]
			class_dict_count[values[8]] += 1
			out_line = str(cnn_class) + ' '
			out_line += str(center_box[0]) + ' ' + str(center_box[1]) + ' '
			out_line += str(width_box) + ' ' + str(height_box) + '\n'
			out.write(out_line)
		out.close()

def getMinMax(mask):
	minMax = [[sys.float_info.max,sys.float_info.max],[-sys.float_info.max,-sys.float_info.max]]
	for coord in mask:
		if coord[0] < minMax[0][0]:
			minMax[0][0] = float(coord[0])
		if coord[0] > minMax[1][0]:
			minMax[1][0] = float(coord[0])
		if coord[1] < minMax[0][1]:
			minMax[0][1] = float(coord[1])
		if coord[1] > minMax[1][1]:
			minMax[1][1] = float(coord[1])
	return minMax

def createYoloLabelsFromAirBus(file,img_dir,dest_dir):
	label_meta = open(file,'r')
	lines = label_meta.readlines()
	i = 0
	for line in lines:
		i += 1
		if i == 1:
			continue
		parts = line.rsplit(',',1)
		coord_parts = parts[1].rsplit()
		if(len(coord_parts) == 0):
			origin = os.path.join(img_dir,parts[0])
			out = os.path.join(dest_dir,parts[0])
			print(origin + '->' + out)
			os.rename(os.path.join(img_dir,parts[0]),os.path.join(dest_dir,parts[0]))

def createLabelFilesFromBBoxDict(file,dest_dir):
	label_meta = open(file,'r')
	lines = label_meta.readlines()
	for line in lines:
		if(line == ',bbox_list\n'):
			continue
		parts = line.rsplit(',"',1)
		out_filename = os.path.splitext(parts[0])[0] + '.txt'
		out_path = os.path.join(dest_dir,out_filename)
		parts[1] = parts[1].replace("\"\n","")
		parts[1] = parts[1].replace("[","")
		parts[1] = parts[1].replace("]","")
		parts[1] = parts[1].replace(")","")
		parts[1] = parts[1].replace("(","")
		parts[1] = parts[1].replace(" ","")
		out = open(out_path,'w')
		boxes_str = parts[1].rsplit(",")
		boxes = []
		for i in range(int(len(boxes_str)/4)):
			box = [[float(boxes_str[i*4]),float(boxes_str[i*4+1])],[float(boxes_str[i*4+2]),float(boxes_str[i*4+3])]]
			boxes.append(box)
		for box in boxes:
			box[0][0] /= 768
			box[0][1] /= 768
			box[1][0] /= 768
			box[1][1] /= 768
			width = box[1][0] - box[0][0]
			height = box[1][1] - box[0][1]
			center = ((box[1][0] + box[0][0])/2,(box[1][1] + box[0][1])/2)
			out_line = str(class_dict['ship']) + ' '
			out_line += str(center[0]) + ' ' + str(center[1]) + ' '
			out_line += str(width) + ' ' + str(height) + '\n'
			out.write(out_line)

def scaleDOTA(folder,label_dir,destination,scale_origin,algo):
	for filename in os.listdir(folder):
		print('scaling ' + filename)
		id = os.path.splitext(filename)[0] + '.txt'
		label = os.path.join(label_dir,id)
		scale_str = open(label).readlines()[1].rsplit('gsd:',1)[1]
		scaling_factor = 1
		if scale_str != 'null\n':
			scaling_factor = float(scale_str)/scale_origin
		img = cv2.imread(os.path.join(folder,filename))
		height,width,channels = img.shape
		resized = cv2.resize(img,(int(width*scaling_factor + 1),int(height*scaling_factor +1)),algo)
		cv2.imwrite(os.path.join(destination,filename),resized)

def scale_dataset(folder,destination,scaling_factor,algo):
	for filename in os.listdir(folder):
		print('scaling ' + filename)
		img = cv2.imread(os.path.join(folder,filename))
		height,width,channels = img.shape
		resized = cv2.resize(img,(int(width*scaling_factor + 1),int(height*scaling_factor +1)),algo)
		cv2.imwrite(os.path.join(destination,filename),resized)


def moveAlternating(folder,out1,out2):
	i = 0
	for filename in os.listdir(folder):
		if i % 2:
			os.rename(os.path.join(folder,filename),os.path.join(out1,filename))
		else:
			os.rename(os.path.join(folder,filename),os.path.join(out2,filename))
		i+=1

def scaleLabel(label_path,out_path,in_w,in_h,w_border,h_border):
	label = open(label_path,'r')
	lines = label.readlines()
	out_lines = ''
	if len(lines) == 0:
		return True
	elif w_border == 0 and h_border == 0:
		return False
	conversion_ratio = [1.0,1.0]
	if w_border != 0:
		conversion_ratio[0] = float(in_w)/float(in_w+w_border)
	elif h_border != 0:
		conversion_ratio[1] = float(in_h)/float(in_h+h_border)
	for line in lines:
		values = line.rsplit()
		loc = [float(values[1]),float(values[2])]
		size = [float(values[3]),float(values[4])]
		for i in range(2):
			loc[i] *= conversion_ratio[i]
			size[i] *= conversion_ratio[i]
		out_lines += values[0] + ' '
		out_lines += str(loc[0]) + ' '
		out_lines += str(loc[1]) + ' '
		out_lines += str(size[0]) + ' '
		out_lines += str(size[1]) + '\n'
	label.close()
	label_out = open(out_path,'w')
	label_out.write(out_lines)
	label_out.close()
	return False

def makeSameSize(folder,dest,labels,labels_dest,hw,algo):
	minimum = [10000,10000]
	maximum = [0,0]
	shapes = []
	i = 0

	for filename in os.listdir(folder):
		print(filename)
		label_path = labels + '/' + os.path.splitext(filename)[0] + '.txt'
		out_label_path = labels_dest  + '/' + os.path.splitext(filename)[0] + '.txt'
		out = dest + '/' + filename
		img = cv2.imread(os.path.join(folder,filename))
		height,width,channels = img.shape
		remove = False
		border = abs(height - width)
		if img.shape not in shapes:
			shapes.append(img.shape)
		if height > width:
			img = cv2.copyMakeBorder(img,0,0,0,border,cv2.BORDER_CONSTANT,0)
			remove = scaleLabel(label_path,out_label_path,width,height,border,0)
		elif width > height:
			img = cv2.copyMakeBorder(img,0,border,0,0,cv2.BORDER_CONSTANT,0)
			remove = scaleLabel(label_path,out_label_path,width,height,0,border)
		elif not remove:
			copyfile(label_path,out_label_path)
		elif remove:
			i += 1
			print(filename)
			os.remove(label_path)
			os.remove(os.path.join(folder,filename))
			continue
		resized = cv2.resize(img,(hw,hw),algo)
		cv2.imwrite(os.path.join(dest,filename),resized)
	print(i)

def writeFileList(folder,outfile):
	out = open(outfile,'w')
	outfolder = 'data/custom/images/'
	for filename in os.listdir(folder):
		#print(os.path.join(outfolder,filename))
		out.write(os.path.join(outfolder,filename) + '\n')
	out.close()

def get_args():
	parser = argparse.ArgumentParser(description='Prep Satellite Imagery Datasets for object detection',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', '--input', metavar='IN', type=str, required=True, help='input path', dest='input')
	parser.add_argument('-d', '--destination', metavar='D', type=str, required=False,help='destination path', dest='dest')
	parser.add_argument('-l', '--labels', metavar='S', type=str, default='', help='dota labels', dest='labels')
	parser.add_argument('-ld', '--labelsdest', metavar='LD', type=str, required=False,help='labels destination path', dest='labeldest')
	parser.add_argument('-s', '--scale', metavar='S', type=float, required=False, help='ratio of input size to output size', dest='scale')
	parser.add_argument('-r', '--resize', metavar='R', type=int, required=False, help='size of output image', dest='resize')
	parser.add_argument('-a', '--algorithm', metavar='A', type=int, default=1, help='interpolation algorith for opencv.resize()', dest='algo')
	return parser.parse_args()

def remove_withoutmask(masks,imgs):
	i = 0
	for filename in os.listdir(imgs):
		mask_file = glob(masks + '/' + filename)
		if(len(mask_file) != 1):
			print(imgs + '/' + filename + " " + masks + '/' + filename)
			os.remove(imgs + '/' + filename)
		else:
			i += 1

def writeValidTrainTxt(input,labels,dest,fracVal):
	classes = [[],[],[],[],[],[],[],[]]
	for filename in os.listdir(input):
		label_path = labels + '/' + os.path.splitext(filename)[0] + '.txt'
		label_info = open(label_path,'r')
		lines = label_info.readlines()
		for line in lines:
			values = line.rsplit()
			classes[int(values[0])].append('data/custom/images/' + filename + '\n')
		label_info.close()
	class_len = [len(classes[0]),len(classes[1]),len(classes[2]),len(classes[3]),len(classes[4]),len(classes[5]),len(classes[6]),len(classes[7])]
	valid_lst =  []
	train_lst =  []
	print(class_len)
	for c in classes:
		random.shuffle(c)
		f = 0
		for file in c:
			if f < int(fracVal*len(c)) and file not in valid_lst:
				valid_lst.append(file)
			elif c not in train_lst:
				train_lst.append(file)
			f += 1
	train_path = dest + '/' + 'train.txt'
	val_path = dest + '/' + 'valid.txt'
	train = open(train_path,'w')
	for filename in train_lst:
		train.write(filename)
	train.close()
	valid = open(val_path,'w')
	for filename in val_list:
		valid.write(filename)
	valid.close()


if __name__ == '__main__':
	args = get_args()
	#assert args.scale > 0.0, 'scaling factor must be greater than 0'
	#assert args.algo < 5, 'algo must be below 5'
	#remove_withoutmask(args.labels,args.input)
	#writeValidTrainTxt(args.input,args.dest,args.scale)
	makeSameSize(args.input,args.dest,args.labels,args.labeldest,args.resize,interpolation_algos[args.algo])
	#writeValidTrainTxt(args.input,args.labels,args.dest,args.scale)
	#writeFileList(args.input,args.dest)
	#moveAlternating(args.input,args.dest,args.labels)
	#createLabelFilesFromBBoxDict(args.input,args.dest)
	#createYoloLabelsFromAirBus(args.labels,args.input,args.dest)
	#scaleDOTALabel(args.input,args.labels,args.dest)
	#scaleDOTA(args.input,args.labels,args.dest,args.scale,interpolation_algos[args.algo])
	#scale_dataset(args.input,args.dest,args.scale,interpolation_algos[args.algo])
