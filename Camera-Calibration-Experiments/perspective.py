import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def perspective_transform(img):
    y, x, _ = img.shape
    # Get object points. A 2D matrix of all flat coordinates
    objp = np.zeros((6*8, 3), np.float32) # 3 channels
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    _, imgpoints = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (8,6), None)
    
    # Get undistort image
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (x,y), None, None)    
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Get image points. Detect from the chessboard
    retCB, corners = cv2.findChessboardCorners(cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY), (8,6), None)
    if retCB:
        cv2.drawChessboardCorners(undistort, (8,6), corners, retCB)

    # From the corners, get the 4 corners from image board. Top left, top right, bottom right, bottom left.
    src = np.float32([corners[0], corners[8-1], corners[-1], corners[-8]])
    # Get the destination coordinates
    offset = 100
    dst = np.float32([[offset, offset], [x-offset, offset], [x-offset, y-offset], [offset, y-offset]])
    
    # Get the transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(undistort, M, (x,y))
    return transformed

dst_img = perspective_transform(img)
plt.figure()
plt.imshow(dst_img)
plt.show()