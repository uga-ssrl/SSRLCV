import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.path import Path
import sys, getopt

def main(argv):
    
    print('=========================')
    print('=  Matches  Visualizer  =')
    print('=========================')
    print('\n^C to quit...')

    match_data_src = 'data/img/multiview_test/matches.txt'
    image0 = 'data/img/multiview_test/Lenna_512(0).png'
    image1 = 'data/img/multiview_test/Lenna_512(1).png'
    image2 = 'data/img/multiview_test/Lenna_512(2).png'
    image3 = 'data/img/multiview_test/Lenna_512(3).png'
    numMatches = 10

    img_left_up  = mpimg.imread(image0)
    img_right_up = mpimg.imread(image1)
    img_right_down  = mpimg.imread(image2)
    img_left_down = mpimg.imread(image3)

    match_data = []

    with open(match_data_src, "r") as f:
      reader = csv.reader(f, delimiter=",")
      for i, line in enumerate(reader):
        match_data.append(line)

    fig = plt.figure()
    a1 = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img_left_up)

    a2 = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(img_right_up)
    
    a3 = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(img_right_down)

    a4 = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(img_left_down)


    
    print('total matches: ' + str(len(match_data)))

    # just get a few matches to show
    s_size = numMatches
    r = len(match_data)
    sample = np.random.randint(r, size=s_size)
    for m_i in sample:
      i = 0
      print('keypoints in match: ' + str(int(match_data[m_i][0])))
      numKP = int(match_data[m_i][0])
      verts = []
      left_up = ()
      right_up = ()
      right_down = ()
      left_down = ()
      while(i < numKP):
        #codes = []
        if(int(match_data[m_i][i*3 + 1]) == 0):
          left_up  = (float(match_data[m_i][i*3+2]), float(match_data[m_i][i*3+3]))
          a1.plot(left_up[0],left_up[1],'ro',markersize=5)
          verts.append('left_up')
          print('left up: ' + str(left_up))
        elif(int(match_data[m_i][i*3+1]) == 1):
          right_up = (float(match_data[m_i][i*3+2]), float(match_data[m_i][i*3+3]))
          a2.plot(right_up[0],right_up[1],'ro',markersize=5)
          verts.append('right_up')
          print('right up: ' + str(right_up))
        elif(int(match_data[m_i][i*3+1]) == 2):
          right_down = (float(match_data[m_i][i*3+2]), float(match_data[m_i][i*3+3]))
          a3.plot(right_down[0],right_down[1],'ro-',markersize=5)
          verts.append('right_down')
          print('right down: ' + str(right_down))
        else:
          left_down = (float(match_data[m_i][i*3+2]), float(match_data[m_i][i+3]))
          a4.plot(left_down[0],left_down[1],'ro',markersize=5)
          verts.append('left_down')
          print('left down: ' + str(left_down))
        i += 1
      for v in range(numKP - 1):
        if(verts[v] == 'left_up'):
          if(verts[v+1] == 'right_up'):
            con = patches.ConnectionPatch(xyA=left_up, xyB=right_up, coordsA="data", coordsB="data", axesA=a1, axesB=a2, color="red")
            a2.add_artist(con)
          elif(verts[v+1] == 'right_down'):
            con = patches.ConnectionPatch(xyA=left_up, xyB=right_down, coordsA="data", coordsB="data", axesA=a1, axesB=a3, color="red")
            a3.add_artist(con)
          else:  
            con = patches.ConnectionPatch(xyA=left_up, xyB=left_down, coordsA="data", coordsB="data", axesA=a1, axesB=a4, color="red")
            a4.add_artist(con)
        elif(verts[v] == 'right_up'):
          if(verts[v+1] == 'right_down'):
            con = patches.ConnectionPatch(xyA=right_up, xyB=right_down, coordsA="data", coordsB="data", axesA=a2, axesB=a3, color="red")
            a3.add_artist(con)
          else:  
            con = patches.ConnectionPatch(xyA=right_up, xyB=left_down, coordsA="data", coordsB="data", axesA=a2, axesB=a4, color="red")
            a4.add_artist(con)
        elif(verts[v] == 'right_down'):
          con = patches.ConnectionPatch(xyA=right_down, xyB=left_down, coordsA="data", coordsB="data", axesA=a3, axesB=a4, color="red")
          a4.add_artist(con)
      if(verts[numKP - 1] == 'right_down' and verts[0] == 'left_up'):
        con = patches.ConnectionPatch(xyA=right_down, xyB=left_up, coordsA="data", coordsB="data", axesA=a3, axesB=a1, color="red")
        a3.add_artist(con)
      elif(verts[numKP - 1] == 'left_down'):
        if(verts[0] == 'left_up'):
          con = patches.ConnectionPatch(xyA=left_down, xyB=left_up, coordsA="data", coordsB="data", axesA=a4, axesB=a1, color="red")
          a4.add_artist(con)
        elif(verts[0] == 'right_up'):
          con = patches.ConnectionPatch(xyA=left_down, xyB=right_up, coordsA="data", coordsB="data", axesA=a4, axesB=a2, color="red")
          a4.add_artist(con)
    plt.show()
if __name__ == "__main__":
    main(sys.argv[1:])