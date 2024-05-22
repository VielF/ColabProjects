import numpy as np
import cv2
from tqdm import tqdm
from queue import Queue


def checkPoint(img, point, visited):
   #logic to detect if a point was visisted
   if ?:
       if visited[point[0], point[1]] == 1: #
           return 0
       return 1
   return 0

def updatePixel(img, point, tolerance, result, average, count):
   B = ? # use the absolute value o img in point (band 0) - average index 0 dividade by count 
   G = ? # use the absolute value o img in point (band 1) - average index 1 dividade by count 
   R = ? # use the absolute value o img in point (band 2) - average index 2 dividade by count 
   
   if B < tolerance and G < tolerance and R < tolerance:
       result[point[0], point[1], 0] = ? # receive img in point, but band 0 
       result[point[0], point[1], 1] = ? # receive img in point, but band 1 
       result[point[0], point[1], 2] = ? # receive img in point, but band 2 
       average[0] += ? # receive imge in point, but band 0 - use float cast 
       average[1] += ? # receive imge in point, but band 1 - use float cast 
       average[2] += ? # receive imge in point, but band 2 - use float cast 
       count += 1
       return 1, result, average, count
   return 0, result, average, count

def ShowResults(filename, result):
   cv2.imwrite(filename, result)


def SeedPointSegmentation(img, seedPoint, tolerance, imgNameOut="out_seed_seg.png"):
   Tolerance = ?? # normalize the value of tolerance to limit of pixel value (eg. 8b is 255) 
   Visited = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
   Result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
   Average = [float(img[seedPoint[0], seedPoint[1], 0]), float(img[seedPoint[0], seedPoint[1], 1]), float(img[seedPoint[0], seedPoint[1], 2])]
   Count = 1

   Q = Queue()
   Q.put([seedPoint[0], seedPoint[1]])
   Visited[seedPoint[0], seedPoint[1]] = 1

   while( Q.qsize() > 0 ):
      # print(Q.qsize())
      CurrentPoint = Q.get()
      Ret, Result, Average, Count  = updatePixel(img, CurrentPoint, Tolerance, Result, Average, Count)
      if Ret == 0:
          continue

      p1 = ? #receive a list with [current point index 0 + 1, current point index 1] 
      p2 = ? #receive a list with [current point index 0 - 1, current point index 1] 
      p3 = ? #receive a list with [current point index 0, current point index 1 - 1] 
      p4 = ? #receive a list with [current point index 0 , current point index 1 + 1] 

      if checkPoint(img, p1, Visited):
         Q.put(p1)
         Visited[p1[0], p1[1]] = 1
      if checkPoint(img, p2, Visited):
         Q.put(p2)
         Visited[p2[0], p2[1]]= 1
      if checkPoint(img, p3, Visited):
         Q.put(p3)
         Visited[p3[0], p3[1]]= 1
      if checkPoint(img, p4, Visited):
         Q.put(p4)
         Visited[p4[0], p4[1]] = 1
   
   ShowResults(imgNameOut, Result)
        

# Image Reading 
ImageList = ["face1.jpg", "face2.jpg", "face3.jpg", "face4.jpg", "colors.jpg"]

Image = cv2.imread(ImageList[1])


# Seed based segmentation 
#seedPoint = [600, 950]
# seedPoint = [250, 250]
# seedPoint = [200, 350]
seedPoint = [120, 150]
SeedPointSegmentation(Image, seedPoint, 30)
