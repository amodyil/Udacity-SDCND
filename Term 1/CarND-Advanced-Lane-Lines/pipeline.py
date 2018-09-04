import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os
import sys
from moviepy.editor import VideoFileClip

### Camera Calibration with OpenCV - Begin  ###
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('camera_cal/calibration*.jpg')

for iname in images:
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)      

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #plt.imsave(os.path.splitext(iname)[0] + "-corners.jpg",  img[...,::-1])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
### Camera Calibration with OpenCV - End  ###

### Test Calibration ###
image_file = 'camera_cal/calibration1.jpg'
orig_img = plt.imread(image_file)
undist = cv2.undistort(orig_img, mtx, dist, None, mtx)
#plt.imsave(os.path.splitext(image_file)[0] + "-original.jpg", orig_img)
#plt.imsave(os.path.splitext(image_file)[0] + "-undistorted.jpg", undist)   


""" Undistort image """
def undistor(image):
    return cv2.undistort(image, mtx, dist, None, mtx)

""" Calculate directional gradient """
def abs_sobel_thresh(image, orient='x', thresh=(0, 255)):  
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

""" Convert to binary image """
def convert_to_thresholded_binary(image, r_thresh=(220, 255), s_thresh=(120, 255), sx_thresh=(20, 100)):   
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    # Threshold x gradient      
    gradient_binary = abs_sobel_thresh(hls[:,:,1], orient='x', thresh=sx_thresh)
    # Threshold color channel        
    R = image[:,:,0] 
    R = cv2.createBackgroundSubtractorMOG2(128,cv2.THRESH_BINARY,1).apply(R)
    R[R==127]=0
    r_binary = np.zeros_like(R)
    r_binary[(R > r_thresh[0]) & (R <= r_thresh[1])] = 1   
    S = hls[:,:,2]
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1
    color_binary = np.zeros_like(s_binary)
    color_binary[(r_binary == 1) | (s_binary == 1)] = 1
    # Combine the thresholds       
    combined_binary = np.zeros_like(gradient_binary)
    combined_binary[(color_binary == 1) | (gradient_binary == 1)] = 1
    return combined_binary

def get_perspective_points(img_size):
    offset = 350 # offset for dst points     
    src_pts = np.float32([[615, 450], 
                         [680, 450], 
                         [img_size[0]-300, 680],
                         [380, 680]])      
    dst_pts = np.float32([[offset, 0], 
                         [img_size[0]-offset, 0], 
                         [img_size[0]-offset, img_size[1]], 
                         [offset, img_size[1]]])                      
    return src_pts, dst_pts

def warper(image):
    img_size = (image.shape[1], image.shape[0])
    src, dst = get_perspective_points(img_size)    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)    
    warped = cv2.warpPerspective(image, M, img_size)
    return warped, Minv

# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):   
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions      
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template   
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    window_centroids.append((l_center,r_center))  
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        # Find the best left centroid by using past left center as a reference        
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))       
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset      
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        window_centroids.append((l_center,r_center))
    return window_centroids

def draw_window_centroids(warped, window_centroids, window_width, window_height, margin):
    if len(window_centroids) > 0:
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
            
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        out_img = np.dstack((warped, warped, warped))*255
        warpage = np.array(out_img,np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)), np.uint8)
    return output

# to cover same y-range as image
ploty = np.linspace(0, 719, num=9)

def curvature(leftx, rightx):
    leftx = np.asarray(leftx[::-1])  # Reverse to match top-to-bottom in y
    rightx = np.asarray(rightx[::-1])  # Reverse to match top-to-bottom in y   
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    leftx_int = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    rightx_int = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]
    position = ((rightx_int+leftx_int)/2)-50
    center = ((rightx_int+leftx_int)/2) - 640
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return position, center, leftx, rightx, left_fitx, right_fitx, left_curverad, right_curverad    

def warpBack(image, warp, persp_Minv, le_fitx, ri_fitx):
    warp_zero = np.zeros_like(warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([le_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([ri_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, persp_Minv, (image.shape[1], image.shape[0])) 
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)
 
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def pipeline(image):  
    undistorted = undistor(image)
    binary = convert_to_thresholded_binary(undistorted)
    warped, MInv = warper(binary)
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)  
    leftX = list(zip(*window_centroids))[0]
    rightX = list(zip(*window_centroids))[1]
    position, center, lx, rx, left_lane.recent_xfitted, right_lane.recent_xfitted, left_lane.radius_of_curvature, right_lane.radius_of_curvature = curvature(leftX, rightX)

    alpha = 0.5
    if left_lane.bestx == None:
        left_lane.bestx = left_lane.recent_xfitted
    else :    
        left_lane.recent_xfitted = left_lane.recent_xfitted * ( 1 - alpha) + alpha * left_lane.bestx
        
    if right_lane.bestx == None:
        right_lane.bestx = right_lane.recent_xfitted
    else :    
        right_lane.recent_xfitted = right_lane.recent_xfitted * ( 1 - alpha) + alpha * right_lane.bestx  
            
    left_lane.detected = False
    right_lane.detected = False 

    max_distance = 0  
    min_distance = 10000

    for i in range(len(left_lane.recent_xfitted)) : 
        point_distance = right_lane.recent_xfitted[i] - left_lane.recent_xfitted[i]

        if point_distance > max_distance:
            max_distance = point_distance
        if point_distance < min_distance:
            min_distance = point_distance    
        
    if (min_distance > 710) and (max_distance < 900):
        left_lane.detected = True
        right_lane.detected = True              
        
    if not left_lane.detected:
        left_lane.recent_xfitted = left_lane.bestx
    else:
        left_lane.bestx = left_lane.recent_xfitted

    if not right_lane.detected:
        right_lane.recent_xfitted = right_lane.bestx
    else:
        right_lane.bestx = right_lane.recent_xfitted
    
    result = warpBack(image, warped, MInv, left_lane.recent_xfitted, right_lane.recent_xfitted)    
    
   # Annotate image with lane curvature estimates
    cv2.putText(result, "L. Curvature: %.2f km" % (left_lane.radius_of_curvature/1000), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    cv2.putText(result, "R. Curvature: %.2f km" % (right_lane.radius_of_curvature/1000), (50,80), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    # Annotate image with position estimate
    cv2.putText(result, "C. Position: %.2f m" % (center*3.7/700), (50,110), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

    return result

left_lane = Line()
right_lane = Line()

def test_pipeline():
    run_test = False
    if run_test == True:
        test_file = 'test_images/test4.jpg' # Test image file
        test_img = plt.imread(test_file)
        
        # Test undistor
        undistorted = undistor(test_img)
        plt.imsave(os.path.splitext(test_file)[0] + "-undistorted.jpg", undistorted)

        # Test convert_to_thresholded_binary 
        binary = convert_to_thresholded_binary(undistorted)
        plt.imsave(os.path.splitext(test_file)[0] + "-binary.jpg", binary,  cmap='gray') 
    
        # Test perspective transform
        warped, perspective_Minv = warper(binary)
        plt.imsave(os.path.splitext(test_file)[0] + "-warped.jpg", warped,  cmap='gray')

        # Test lane finding
        window_centroids = find_window_centroids(warped, window_width, window_height, margin)    
        fitted = draw_window_centroids(warped, window_centroids, window_width, window_height, margin)
        plt.imsave(os.path.splitext(test_file)[0] + "-fitted.jpg", fitted,  cmap='gray')

        # Test lane curvature
        leftx = list(zip(*window_centroids))[0]
        rightx = list(zip(*window_centroids))[1]
        position, center, leftx, rightx, left_fitx, right_fitx, left_curverad, right_curverad = curvature(leftx, rightx)
        mark_size = 3
        fig = plt.figure()    
        plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis()
        print(left_curverad, 'm', right_curverad, 'm')
        fig.savefig(os.path.splitext(test_file)[0] + "-curved.jpg")

        # Test wrap back
        warpbacked = warpBack(test_img, warped, perspective_Minv, left_fitx, right_fitx)
        plt.imsave(os.path.splitext(test_file)[0] + "-warpedback.jpg", warpbacked)

        # Test final pipeline   
        plt.imsave(os.path.splitext(test_file)[0] + "-final.jpg", pipeline(test_img))

def apply_pipeline_to_video():
    # Apply pipeline on project video stream 
    white_output = 'project_video_final.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(white_output, audio=False)

def main(arguments):
    test_pipeline()
    apply_pipeline_to_video()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))