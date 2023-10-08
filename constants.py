# feature descriptor constants
COLOR_MOMENTS = "color_moments"
HOG = "hog"
ResNet_AvgPool_1024 = "resnet_avgpool_1024"
ResNet_Layer3_1024 = "resnet_layer3_1024"
ResNet_FC_1000 = "resnet_fc_1000"

# file path constants
IMAGES_SOURCE_DIR = "./Images/images-caltech"
IMAGES_DEST_DIR = "./Images/all-images/"
RESIZED_IMAGES_DEST_DIR = "./Images/resized-images/"
OUTPUT_DEST_DIR = "./Output/"
OUTPUT_FD_FILENAME_DIR = "./Output/%s/%s.csv"
RESIZED_IMAGES_DEST_DIR_FORMAT = "./Images/resized-images/%s/*"

# similarity measure constants
L1_NORM = "l1_norm"
L2_NORM = "l2_norm"
L_MAX = "l_max"
COSINE_SIMILARITY = "cosine_similarity"
INTERSECTION = "intersection"

# other constants
FD_DELIMITER = "|"
