# libraries
import os
import glob
import pathlib
from PIL import Image
import cv2
import numpy
from scipy.stats import skew
from pandas import DataFrame

# internal packages
import constants
import resnet
import utils

# global variables
resnet_fd_values = {
    constants.ResNet_AvgPool_1024: {},
    constants.ResNet_Layer3_1024: {},
    constants.ResNet_FC_1000: {},
}
RESNET_FDs = [constants.ResNet_AvgPool_1024, constants.ResNet_Layer3_1024, constants.ResNet_FC_1000]
resnet_model = None

# lambda functions
get_resized_image_dest_path = lambda resize, mode: constants.RESIZED_IMAGES_DEST_DIR_FORMAT%("%d_%d_%s"%(resize[0], resize[1], mode))

# helper functions
def get_fd_output_filename(fd, image_id):
    if fd in RESNET_FDs:
        return constants.OUTPUT_FD_FILENAME_DIR%(fd, 'output')
    return constants.OUTPUT_FD_FILENAME_DIR%(fd, image_id)

def get_all_image_ids():
    return [os.path.basename(i).split('.')[0] for i in glob.glob(constants.IMAGES_DEST_DIR + "*")]

def get_partition_by_channel(image, channel, partition_size, mode = "rgb"):
    if mode == "rgb":
        channel_mat = image[:,:,channel]
    else:
        channel_mat = image
    total_rows, total_cols = len(channel_mat), len(channel_mat[0])
    n_rows_per_cell, n_cols_per_cell = total_rows//partition_size[0], total_cols//partition_size[1]
    partition_mat = numpy.zeros((partition_size[0], partition_size[1], n_rows_per_cell, n_cols_per_cell))
    a, b = 0, 0
    for i in range(0, total_rows, n_rows_per_cell):
        b = 0
        for j in range(0, total_cols, n_cols_per_cell):
            partition_mat[a][b] = channel_mat[i:i+n_rows_per_cell, j:j+n_cols_per_cell]
            b+=1
        a+=1
    return partition_mat

# saves generated FDs in output files
def save_to_file(output, fd, image_id = ""):
    if fd in RESNET_FDs:
        dest_path = constants.OUTPUT_DEST_DIR + fd
        pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)
        output_file = dest_path + "/output.csv"
        d = {}
        for i in output:
            d[i] = output[i][fd]
        DataFrame(d).T.to_csv(output_file)
    else:
        dest_path = constants.OUTPUT_DEST_DIR + fd
        pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)
        output_file = dest_path + "/%s.csv"%(image_id)

        np_output = numpy.asarray(output)
        numpy.savetxt(output_file, np_output, delimiter = ',', fmt = '%s')

def flatten(mat):
    return numpy.asarray(mat).flatten()

### color moment functions

# calculates a particular color moment - mean, std or skewness. color moment func is passed
def get_color_moment(image, color_moment_func, partition):
    color_moments = [[[0 for i in range(partition[1])] for i in range(partition[0])] for i in range(3)]
    for ch in range(3):
        partition_mat = get_partition_by_channel(image, ch, partition)
        for i in range(partition[0]):
            for j in range(partition[1]):
                color_moments[ch][i][j] = color_moment_func(partition_mat[i][j])
    return color_moments

# calculates all color moments for an image
def get_all_color_moments(image_path, partition):
    color_moments_functions = {"mean": numpy.mean, "standard_deviation": numpy.std, "skew": skewness}
    image = cv2.imread(image_path)
    cm_grid = [[[] for i in range(partition[1])] for i in range(partition[0])]
    for cm_func_name in color_moments_functions:
        cm_val = get_color_moment(image, color_moments_functions[cm_func_name], partition)
        for i in range(partition[0]):
            for j in range(partition[1]):
                for ch in range(3):
                    cm_grid[i][j].append(cm_val[ch][i][j])
    output_grid = []
    for i in range(partition[0]):
        output_grid.append([])
        for j in range(partition[1]):
            output_grid[-1].append(constants.FD_DELIMITER.join([str(k) if not numpy.isnan(k) else '0' for k in cm_grid[i][j]]))

    return output_grid

def skewness(arr):
    return skew(arr.flatten(), nan_policy='omit')

# returns a dictionary of image ID to color moments grid and also saves them in output files
def get_color_moments_of_all_images():
    dest_path = get_resized_image_dest_path((300, 100), "rgb")
    output = {}
    count = 0
    for i in glob.glob(dest_path):
        image_id = os.path.basename(i).split('.')[0]
        output[image_id] = get_all_color_moments(i, (10, 10))
        save_to_file(output[image_id], constants.COLOR_MOMENTS, image_id=image_id)
        print("Done; image - %s; count: %d"%(image_id, count))
        count += 1
    return output

### HOG functions

# returns a dictionary of image ID to HOG grid and also saves them in output files
def get_hog_of_all_images():
    dest_path = get_resized_image_dest_path((300, 100), "l")
    output = {}
    count = 0
    for i in glob.glob(dest_path):
        image_id = os.path.basename(i).split('.')[0]
        output[image_id] = get_all_color_moments(i, (10, 10))
        save_to_file(output[image_id], constants.HOG, image_id=image_id)
        print("Done; image - %s; count: %d"%(image_id, count))
        count += 1
    return output

# calculates HOG values for an image for the passed image path
def hog(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    partitioned_mat = get_partition_by_channel(image, 0, (10, 10), mode = "l")
    result_mat = []
    for i in range(len(partitioned_mat)):
        result_mat.append([])
        for j in range(len(partitioned_mat[0])):
            result_mat[-1].append(constants.FD_DELIMITER.join([str(i) for i in hog_for_partition(partitioned_mat[i][j])]))
    return result_mat

# calculates HOG vector for each partition
def hog_for_partition(mat):
    gx, gy = cv2.Sobel(mat, cv2.CV_64F, 1, 0, ksize=1), cv2.Sobel(mat, cv2.CV_64F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return hog_to_bins(mag, angle)

def hog_to_bins(mag, angle):
    l = [0 for i in range(9)]
    for i in range(len(mag)):
        for j in range(len(mag[0])):
            wb_l, wb_r = weighted_bins(angle[i][j])
            l[wb_l[0].astype(numpy.int64)] += (wb_l[1] * mag[i][j])
            l[wb_r[0].astype(numpy.int64)] += (wb_r[1] * mag[i][j])
    return l

# calculates weighted bins for each angle sector - (0, 40, 80, 120, 160, 200, 240, 280, 320)
def weighted_bins(angle_value):
    left_bin, right_bin = angle_value//40, angle_value//40 + 1
    distance_to_left, distance_to_right = abs(angle_value - left_bin*40), abs(angle_value - right_bin*40)
    percentage_left = distance_to_right/(distance_to_right+distance_to_left)
    percentage_right = 1 - percentage_left
    return [(left_bin, percentage_left), (right_bin%9, percentage_right)]

### Resnet functions

# fetches a vector of size 1024 for avgpool output layer of resnet
def get_ResNet_AvgPool_1024(output_arr):
    return [numpy.mean([output_arr[i].item(), output_arr[i+1].item()]) for i in range(0, 2048, 2)]

# fetches a vector of size 1024 for layer3 output layer of resnet
def get_ResNet_Layer3_1024(output_arr):
    return list(map(lambda x: numpy.mean(x.detach().numpy()), output_arr))

# fetches a vector of size 1024 for avgpool FC layer of resnet
def get_ResNet_FC_1000(output_arr):
    return [i.item() for i in output_arr]

# fetches resnet model; initializes if not already done
def get_resnet():
    global resnet_model
    if resnet_model == None:
        resnet_model = resnet.ResNetWithHooks(['avgpool', 'layer3', 'fc'])
    return resnet_model

# loops through all images to get all resnet FDs
def run_resnet_for_all_images():
    output = {}
    count = 0
    for image_path in glob.glob(get_resized_image_dest_path((224, 224), "rgb")):
        image_id = os.path.basename(image_path).split('.')[0]
        output[image_id] = run_resnet(image_path)
        count += 1
        print("Done - image: %s, count: %d"%(image_id, count))
    return output

# opens image, processes it and fetches all extracted output layers
def run_resnet(image_path):
    img = Image.open(image_path)
    model = get_resnet()
    i = model.preprocess(img)
    model(i)
    resnet_output = model.get_output_layers()
    resnet_features = {
        constants.ResNet_AvgPool_1024: get_ResNet_AvgPool_1024(resnet_output['avgpool'][0]),
        constants.ResNet_Layer3_1024: get_ResNet_Layer3_1024(resnet_output['layer3'][0]),
        constants.ResNet_FC_1000: get_ResNet_FC_1000(resnet_output['fc'][0]),
    }
    return resnet_features

# generates resnet FDs for all images and stores them in CSV output files
def get_resnet_fds_of_all_images():
    data = run_resnet_for_all_images()

    for fd in RESNET_FDs:
        save_to_file(data, fd)

### main FD generator functions

# extracts all images, resizing them into required dimensions and saves them in separate folders
def extract_all_images():
    utils.extract_all_images(use_torchvision=True)
    utils.resize_all_images((300, 100))
    utils.resize_all_images((300, 100), mode = "l")
    utils.resize_all_images((224, 224))
    
# generates_all required feature descriptions - color moments, HOG, resnet avgpool, resnet FC and resnet layer3
def generate_all_fds(skip_extraction = False):
    if not skip_extraction:
        extract_all_images()
    get_color_moments_of_all_images()
    get_hog_of_all_images()
    get_resnet_fds_of_all_images()

# once FDs are generated and stored, this function helps to read each FD file, it's internally used by read_fd_for_image function
def read_fds(fd, filename):
    if fd in RESNET_FDs:
        with open(filename) as file:
            output, is_first_line = {}, True
            for line in file:
                if not is_first_line:
                    values = line.strip().split(',')
                    output[values[0]] = numpy.asarray([float(i) for i in values[1:]])
                is_first_line = False
        resnet_fd_values[fd] = output
        return output
    else:
        with open(filename) as file:
            output = []
            for line in file:
                l = line.strip()
                output.append([])
                for i in l.split(','):
                    output[-1].append([float(j) for j in i.split('|')])
        return output

# this function is used to fetch flattened feature description vector given feature description and image ID
def read_fd_for_image(fd, image_id):
    fd_filename = get_fd_output_filename(fd, image_id)
    if fd in RESNET_FDs:
        global resnet_fd_values
        if image_id not in resnet_fd_values[fd]:
            output = read_fds(fd, fd_filename)
        else:
            output = resnet_fd_values[fd]
        return output[image_id]
    else:
        return flatten(read_fds(fd, fd_filename))
