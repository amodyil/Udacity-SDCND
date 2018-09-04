import numpy as np
import cv2
import glob
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import sys
import collections
from moviepy.editor import VideoFileClip

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {"n_cars":len(car_list), "n_notcars":len(notcar_list)}
    example_img = mpimg.imread(car_list[0])
    data_dict["image_shape"] = example_img.shape
    data_dict["data_type"] = example_img.dtype
    return data_dict

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)          
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)     

def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel() 
                        
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        if color_space != 'RGB':
            feature_image = convert_color(image, conv=color_space)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            file_features.append(bin_spatial(feature_image, size=spatial_size))
        if hist_feat == True:
            file_features.append(color_hist(feature_image, nbins=hist_bins))
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return features   

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=5):
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)  - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    b_boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))     
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 and svc.decision_function(test_features)[0] > 0.2:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                b_boxes.append([(xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)])
                
    return b_boxes

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 5)

    return img

def read_images():
    """Returns vehicles, non_vehicles."""    
    non_vehicles  = glob.glob('data/non-vehicles/*/*.png')
    vehicles = glob.glob('data/vehicles/*/*.png')
    return vehicles, non_vehicles 

def train(vehicle_images, non_vehicle_images):
    color_space = 'RGB2YCrCb'
    orient = 9 
    pix_per_cell = 8 
    cell_per_block = 2 
    hog_channel = 'ALL'
    spatial = (32, 32)
    hist_bins = 32 
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    vehicle_features = extract_features(vehicle_images, color_space=color_space, 
                            spatial_size=spatial, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    non_vehicle_features = extract_features(non_vehicle_images, color_space=color_space, 
                            spatial_size=spatial, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)                        
    # Scaler tp standardize features by removing the mean and scaling to unit variance
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=np.random.randint(0, 100))

    print('Using:',color_space,'colorspace',
                   orient,'orientations',
                   pix_per_cell, 'pixels per cell and', 
                   cell_per_block,'cells per block', 
                   hog_channel, 'hog channel',
                   spatial,'spatial bins',
                   hist_bins, 'hist bins')

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC(C=0.0001)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    return svc, X_scaler 

# Read all vehicles and non-vehicles
vehicles, non_vehicles = read_images()
# Train classifier
svc, X_scaler = train(vehicles, non_vehicles) 

heatmaps = collections.deque(maxlen=50)
previous_labels = []

def pipeline(image): 
    hot_windows = find_cars(image, ystart=400, ystop=650, scale=1.5, svc=svc, X_scaler=X_scaler, 
                                   orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 1)
    heatmaps.append(heat)
    heatmap = np.clip(sum(heatmaps), 0, 255)
    labels = label(heatmap)     
    
    if labels:
        previous_labels = labels 
        final_labels = labels
    else: 
        final_labels = previous_labels
    
    return draw_labeled_bboxes(np.copy(image), final_labels)

def test_pipeline():     
    run_test = False
    if run_test == True:   
        # vehicles and non-vehicles stats
        data_info = data_look(vehicles, non_vehicles)
        print('Your function returned a count of', data_info["n_cars"], ' vehicles and',  data_info["n_notcars"], ' non-vehicles')
        print('of size: ',data_info["image_shape"], ' and data type:', data_info["data_type"])   
        vehicle_image = mpimg.imread(vehicles[ np.random.randint(0, len(vehicles))])
        plt.imsave("output_images/vehicle.png", vehicle_image)
        plt.imsave("output_images/non_vehicle.png", mpimg.imread(non_vehicles[ np.random.randint(0, len(non_vehicles))])) 

        # Test hog features with colorspace = YCrCb, orient = 9, pix_per_cell = 8, cell_per_block = 2
        feature_image = convert_color(vehicle_image, conv='RGB2YCrCb')
        for channel in range(feature_image.shape[2]):
            features, hog_image = get_hog_features(feature_image[:,:,channel], 
                                                   orient = 9, pix_per_cell = 8, cell_per_block = 2, 
                                                   vis=True, feature_vec=True)
            plt.imsave("output_images/hog_channel" + str(channel) +  "_vehicle.png", hog_image, cmap='gray')

        # Test Hog Sub-sampling Window Search with classifier
        test_images = glob.glob('test_images/*.jpg')
        i = 0
        for img in test_images:
            i = i + 1
            image = mpimg.imread(img)
            draw_image = np.copy(image)        
            hot_windows = find_cars(image, ystart=400, ystop=650, scale=1.5, svc=svc, X_scaler=X_scaler, 
                                           orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32)
            plt.imsave("output_images/window_img" + str(i) + ".png", draw_boxes(draw_image, hot_windows))

            # Test heatmaps filters
            heat = np.zeros_like(image[:,:,0]).astype(np.float)
            heat = add_heat(heat, hot_windows)         
            heat = apply_threshold(heat, 0.5)  # Apply threshold to help remove false positives
            heatmap = np.clip(heat, 0, 255)
            labels = label(heatmap)
            plt.imsave("output_images/heatmap" + str(i) +  ".png", heatmap, cmap='hot') 
            plt.imsave("output_images/heatmap_image" + str(i) +  ".png", draw_labeled_bboxes(draw_image, labels))                       

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