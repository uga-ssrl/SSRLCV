import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch
import sys, getopt

def main(argv):
    
    print('=========================')
    print('=  Matches  Visualizer  =')
    print('=========================')
    print('\n^C to quit...')

    match_data_src = ''
    image1 = ''
    image2 = ''
    opts, args = getopt.getopt(argv,"hm:1:2:n",["matches=","image1=","image2=","numshow="])
    numMatches = 0
    for opt, arg in opts:
        if opt in ('-h1','--help'):
            print('usage: -m matches -1 image1 -2 image2')
        if opt in ('-m','--matches'):
            match_data_src = arg
            print(match_data_src)
        elif opt in ('-1','--image1'):
            image1 = arg
            print(image1)
        elif opt in ('-2','--image2'):
            image2 = arg
            print(image2)
        elif opt in ('-n', '--numshow'):
            numMatches = int(arg)

    img_left  = mpimg.imread(image1)
    img_right = mpimg.imread(image2)


    match_data = []

    with open(match_data_src, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            match_data.append(line)

    fig = plt.figure()
    a1 = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img_left)
    a1.set_title('Image 1')

    a2 = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img_right)
    imgplot.set_clim(0.0, 1.0)
    a2.set_title('Image 2')

    print('total matches: ' + str(len(match_data)))

    # just get a few matches to show
    s_size = numMatches
    range = len(match_data)
    sample = np.random.randint(range, size=s_size)
    for m_i in sample:
        left  = (float(match_data[m_i][0]), float(match_data[m_i][1]))
        right = (float(match_data[m_i][2]), float(match_data[m_i][3]))
        # print 'left:'
        # print left
        # print 'right:'
        # print right
        con = ConnectionPatch(xyA=right, xyB=left, coordsA="data", coordsB="data", axesA=a2, axesB=a1, color="red")
        a2.add_artist(con)
        a1.plot( left[0], left[1],'ro',markersize=10)
        a2.plot(right[0],right[1],'ro',markersize=10)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])