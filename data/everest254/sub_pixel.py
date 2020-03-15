################################################
#
# sub_pixel.py
# Author: Nicholas Neel
# Purpose: To test subpixel matches over feature
#	   space given matched feature locations. 
#
################################################

import copy, math, numpy, scipy, cv2
from scipy.interpolate import interpolate
def mag(img,i,j):
	return math.sqrt((img[i+1,j]-img[i-1,j])**2 + (img[i,j+1]-img[i,j-1])**2)
def orientation(img,i,j):
	return math.atan2(img[i,j+1]-img[i,j-1] , img[i+1,j]-img[i-1,j])




def createDescriptor(img,x_loc,y_loc):
	descriptor = numpy.zeros((128))
	blur_Image = cv2.GaussianBlur(img, (7,7), img.std())
	
	blur_Image = blur_Image.astype(numpy.float)
	img = img.astype(numpy.float)

	feature_window = numpy.zeros((16,16)) #17x17. need 16.16
	mag_window = numpy.zeros((16,16))
	o_window = numpy.zeros((16,16))
	for i in range(16):
		for j in range(16):
			temp_val_m = 0
			temp_val_o = 0
			temp_val   = 0
			for avgx in range(3):
				for avgy in range(3):
					if avgx == 1 or avgy == 1:
						continue
					temp_val += img[x_loc+i+(avgx-1)-8,y_loc+(avgy-1)+j-8]
			temp_val = temp_val/4.
			feature_window[i][j] = copy.copy(temp_val)
			mag_window[i,j] = mag(img,x_loc+i-8,y_loc+j-8)
			o_window[i,j] = orientation(img,x_loc+i-8,y_loc+j-8)
	#return feature_window,mag_window,o_window
	
	# Split into list of 4x4 arrays
	arrs = []
	mag_arrs = []
	o_arrs = []
	for i in range(4):
		for j in range(4):
			arrs.append(feature_window[4*i:4*i+4,4*j:4*j+4])
			mag_arrs.append(mag_window[4*i:4*i+4,4*j:4*j+4])
			o_arrs.append(    o_window[4*i:4*i+4,4*j:4*j+4])

	#return arrs,mag_arrs,o_arrs
	#'''
	for d in range(len(arrs)):
		temp_descriptor = [0,0,0,0,0,0,0,0]
		for i in range(4):
			for j in range(4):
				curr_o = o_arrs[d][i,j]%(2*math.pi)
				if   0 <= curr_o and curr_o < math.pi/4:
					temp_descriptor[0] += mag_arrs[d][i,j] 
				elif 1*math.pi/4   <=  curr_o and curr_o  < 2*math.pi/4:
					temp_descriptor[1] +=  mag_arrs[d][i,j]
				elif 2*math.pi/4 <=   curr_o and curr_o  < 3*math.pi/4:
					temp_descriptor[2] +=  mag_arrs[d][i,j]
				elif 3*math.pi/4 <=  curr_o and curr_o  < 4*math.pi/4:
					temp_descriptor[3] +=  mag_arrs[d][i,j]
				elif 4*math.pi/4 <=  curr_o and curr_o   < 5*math.pi/4:
					temp_descriptor[4] +=  mag_arrs[d][i,j]
				elif 5*math.pi/4 <= curr_o  and curr_o  < 6*math.pi/4:
					temp_descriptor[5] +=  mag_arrs[d][i,j]
				elif 6*math.pi/4 <= curr_o  and curr_o  < 7*math.pi/4:
					temp_descriptor[6] +=  mag_arrs[d][i,j]
				elif 7*math.pi/4 <= curr_o  and curr_o  < 8*math.pi/4:
					temp_descriptor[7] +=  mag_arrs[d][i,j]
				else:
					print "ERROR: Orientation not defined"
					print curr_o
		descriptor[8*d:8*d+8] = copy.deepcopy(temp_descriptor)
	return descriptor
	#'''
def coarseMatch(d1,d2):
	return (numpy.inner(d1-d2,d1-d2))**.5
def sadMatch(d1,d2):
	res = 0
	for i in range(len(d1)):
		res += abs(d1[i] - d2[i])
	return res


def readMatches(filename):
	z = open(filename)
	raw = z.read()
	z.close()
	data = raw.split("\n")
	for match in range(len(data)-1):
		data[match] = data[match].split(",")
		for i in range(4):
			data[match][i] = int(data[match][i])
	return data

def featureSpace(img):
	data = numpy.zeros((len(img),len(img[0])),numpy.ndarray)
	for i in range(9,len(img)-9):
		for j in range(9,len(img)-9):
			data[i,j] = createDescriptor(img,i,j)
		print i
	return data

def submatchWindow(feature_space_1,feature_space_2,v):
	m1x=v[0] 
	m1y=v[1]
	m2x=v[2]
	m2y=v[3]
	M1 = numpy.zeros((11,11))
	M2 = numpy.zeros((11,11))

	x  = numpy.arange(-5,5,1)
	y  = numpy.arange(-5,5,1)
	
 
	f1 = feature_space_1[m1x,m1y]
	f2 = feature_space_2[m2x,m2y]
	for i in range(11):
		for j in range(11):
			#M1[i,j] = coarseMatch(f2,feature_space_1[m1x+i-5,m1y+j-5])
			#M2[i,j] = coarseMatch(f1,feature_space_2[m2x+i-5,m2y+j-5])
			M1[i,j] = sadMatch(f2,feature_space_1[m1x+i-5,m1y+j-5])
			M2[i,j] = sadMatch(f1,feature_space_2[m2x+i-5,m2y+j-5])

	sq1 = M1[:10,:10]
	sq2 = M2[:10,:10]

	l_bound = -1.
	u_bound = 1
	step    = 100.
	new_x = numpy.linspace(l_bound,u_bound,step)
	new_y = numpy.linspace(l_bound,u_bound,step)

	#return sq1, sq2
	#'''
	S1 = interpolate.interp2d(x,y,sq1,kind="cubic")
	S2 = interpolate.interp2d(x,y,sq2,kind="cubic")
	new_z1 = S1(new_x,new_y)
	new_z2 = S2(new_x,new_y) 

	#return new_z1, new_z2
		
	sp1x,sp1y = numpy.where(new_z1 == new_z1.min())
	sp2x,sp2y = numpy.where(new_z2 == new_z2.min())
	#print sp1x, sp1y, sp2x, sp2y
	if (sp1x[0] == 0 or sp1x[0] == step-1 or sp1y[0]==0 or sp1y[0]==step-1) and (sp2x[0] == 0 or sp2x[0] == step-1 or sp2y[0] == 0 or sp2y==step-1):
		return [m1x,m1y,m2x,m2y]
	elif sp1x[0] == 0 or sp1x[0] == step-1 or sp1y[0]==0 or sp1y[0]==step-1:
		return [m1x,m1y,m2x+(sp2x[0]/step+l_bound),m2y+(sp2y[0]/step+l_bound)]

	elif sp2x[0] == 0 or sp2x[0] == step-1 or sp2y[0] == 0 or sp2y==step-1:
		return [m1x+(sp1x[0]/step+l_bound),m1y+(sp1y[0]/step+l_bound),m2x,m2y]
	else:
		return [m1x+(sp1x[0]/step+l_bound),m1y+(sp1y[0]/step+l_bound),m2x+(sp2x[0]/step+l_bound),m2y+(sp2y[0]/step+l_bound)]



def submatchError(feature_space_1,feature_space_2,v):
	m1x=v[0] 
	m1y=v[1]
	m2x=v[2]
	m2y=v[3]
	M1 = numpy.zeros((11,11))
	M2 = numpy.zeros((11,11))

	x  = numpy.arange(-5,5,1)
	y  = numpy.arange(-5,5,1)
	
 
	f1 = feature_space_1[m1x,m1y]
	f2 = feature_space_2[m2x,m2y]
	for i in range(11):
		for j in range(11):
			M1[i,j] = coarseMatch(f2,feature_space_1[m1x+i-5,m1y+j-5])
			M2[i,j] = coarseMatch(f1,feature_space_2[m2x+i-5,m2y+j-5])

	
	sq1 = M1[:10,:10]
	sq2 = M2[:10,:10]

	l_bound = -1.
	u_bound = 1
	step    = 100.
	new_x = numpy.linspace(l_bound,u_bound,step)
	new_y = numpy.linspace(l_bound,u_bound,step)

	#return sq1, sq2
	#'''
	S1 = interpolate.interp2d(x,y,sq1,kind="cubic")
	S2 = interpolate.interp2d(x,y,sq2,kind="cubic")
	new_z1 = S1(new_x,new_y)
	new_z2 = S2(new_x,new_y) 

	#return new_z1, new_z2
		
	sp1x,sp1y = numpy.where(new_z1 == new_z1.min())
	sp2x,sp2y = numpy.where(new_z2 == new_z2.min())
	return new_z2.min()

def finalError(fs1,fs2,matches):
	lst = []
	for a in matches:
		lst.append(submatchError(fs1,fs2,a))
	return lst

def final(fs1, fs2, matches):
	counter = 0
	errors  = 0
	lst     = []
	print len(matches)
	for a in matches:
		try:
			lst.append(submatchWindow(fs1,fs2,a))
		except:
			errors += 1
		counter += 1
		if counter%(len(matches)/10) == 0:
			print "Error rate: ", errors/float(counter)*100
	return lst
	
	
