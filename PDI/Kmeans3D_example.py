import numpy as np
import cv2
import sys
from random import randint as randi
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def bgr2hex(bgr):
   return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def ScatterPlot(img, centroids, clutserLabels, plotNameOut="scatterPlot.png"):
    fig = plt.figure()
    ax = Axes3D(fig)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            ax.scatter(img[x, y, 2], img[x, y, 1], img[x, y, 0], color = bgr2hex(centroids[clutserLabels[x, y]]))
    plt.show()
    plt.savefig(plotNameOut)

def ShowCluster(img, centroids, clusterLabels, imgNameOut="out.png"):
   result = np.zeros((img.shape), dtype=np.uint8)
   for i in range(img.shape[0]):
       for j in range(img.shape[1]):
           bgr = centroids[clusterLabels[i, j]]
           result[i, j, 0] = np.uint8(bgr[0])
           result[i, j, 1] = np.uint8(bgr[1])
           result[i, j, 2] = np.uint8(bgr[2])
   cv2.imwrite(imgNameOut, result)
   #some problem - it's not necessary
   #TODO
   ScatterPlot(img, centroids, clusterLabels, plotNameOut="scatterPlot.png")
   cv2.imshow("K-Mean Cluster", result)
   cv2.waitKey(0)

def GetEuclideanDistance(Cbgr, Ibgr):
   b = float(Cbgr[0]) - float(Ibgr[0])
   g = float(Cbgr[1]) - float(Ibgr[1])
   r = float(Cbgr[2]) - float(Ibgr[2])
   return sqrt(b*b + g*g + r*r)

def KMeans3D(img, k=2, max_iterations=100, imgNameOut="out.png"):
   '''
   KMeans algorithm base to 3D data image (RGB)
   input: 
      img: image open using a library like OpenCV
      k: number of segments
      max_iterations: number max of interations to segment image 
   '''
   Clusters = k
   # create the centroids of algorithm - each k is there is a center called "centroid"
   centroids = np.zeros((k, 3), dtype=np.float64)  #for 5D, create a matrix of zeros with (k, 5)
   for i in range(Clusters):
      #start get the initial point of segmentation in X and Y coordinates 
      x = randi(0, img.shape[0]-1)
      y = randi(0, img.shape[1]-1)
      #get the RGB (BGR) from img and divide in different matrix
      b = float(img[x, y, 0])
      g = float(img[x, y, 1])
      r = float(img[x, y, 2])

      centroids[i, 0] = b
      centroids[i, 1] = g
      centroids[i, 2] = r
      #centroids[i, 3] = x
      #centroids[i, 4] = y
      #in 5D add too x and y in centroids


   print("Centroids:\n", centroids)
   ClusterLabels = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
   for i in range(max_iterations):
       for x in range(img.shape[0]):
           for y in range(img.shape[1]):
               MinDist = sys.float_info.max
               for c in range(Clusters):
                   dist = GetEuclideanDistance(centroids[c], img[x, y])
                   if dist <= MinDist:
                       MinDist = dist
                       ClusterLabels[x, y] = c

       # update Mean of the clusters
       MeanCluster = np.zeros((Clusters, 4), dtype=np.float64)
       for x in range(img.shape[0]):
           for y in range(img.shape[1]):
               clusterNumber = ClusterLabels[x, y]
               MeanCluster[clusterNumber, 0] += 1
               MeanCluster[clusterNumber, 1] += float(img[x, y, 0])
               MeanCluster[clusterNumber, 2] += float(img[x, y, 1])
               MeanCluster[clusterNumber, 3] += float(img[x, y, 2])
               #in 5D add too float(x) and float(y) in centroids
       
       copy = np.copy(centroids)
       for c in range(Clusters):
           # print("MeanCluster["+ str(c) +", 0]:", MeanCluster[c, 0])
           centroids[c, 0] = MeanCluster[c, 1] / MeanCluster[c, 0]
           centroids[c, 1] = MeanCluster[c, 2] / MeanCluster[c, 0]
           centroids[c, 2] = MeanCluster[c, 3] / MeanCluster[c, 0]
           #in 5D add too x (4) and y (5) in centroids

       Same = True
       for i in range(centroids.shape[0]):
           for j in range(centroids.shape[1]):
               if copy[i, j] != centroids[i, j]:
                   Same = False
                   break
           if not Same:
               break
       if Same:
           break
   ShowCluster(img, centroids, ClusterLabels, imgNameOut)

ImageNames = ["2apples.jpg", "2or4objects.jpg", "colors.jpg"]
No = 1
Image = cv2.imread(ImageNames[No])
Image = cv2.resize(Image, None, fx=0.25, fy=0.25)
print("Image Size:", Image.shape)


KMeans3D(Image, k=2, max_iterations=10, imgNameOut="img_out.png")



