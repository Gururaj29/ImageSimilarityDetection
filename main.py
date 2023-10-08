import feature_descriptors
import similarity_measures
import constants

# input variables
query_image_id = "8676"
k = 4 # k most similar images
fd = constants.ResNet_Layer3_1024 # feature descriptor to be used
similarity_measure = constants.COSINE_SIMILARITY # similarity measure to be used

# generates all feature descriptions and stores in output file, comment out if they are already generated
feature_descriptors.generate_all_fds()

# shows k most similar iamges
similarity_measures.show_k_most_similar_images(query_image_id, k, fd, similarity_measure)