# libraries
import numpy

# internal packages
import feature_descriptors
import constants
import utils

# l1 norm distance measure, since we want similarity measure output is the negation of the l1 norm
def l1_norm(v1, v2):
    return (sum([abs(v1[i]-v2[i]) for i in range(len(v1))])) * -1

# l2 norm distance measure, since we want similarity measure output is the negation of the l2 norm
def l2_norm(v1, v2):
    return (numpy.linalg.norm(numpy.array(v1) - numpy.array(v2))) * -1

# l max norm distance measure, since we want similarity measure output is the negation of the l max norm
def l_max(v1, v2):
    return (max([abs(v1[i]-v2[i]) for i in range(len(v1))])) * -1

# intersection similarity measure
def intersection(v1, v2):
    min_count, max_count = 0, 0
    for i in range(len(v1)):
        min_count += min(v1[i], v2[i])
        max_count += max(v1[i], v2[i])
    return min_count/max_count

# cosine similarity measure
def cosine_similarity(v1, v2):
    return numpy.dot(v1,v2)/(numpy.linalg.norm(v1)*numpy.linalg.norm(v2))

# map of similarity measure name to the function
similarity_function_map = {
    constants.L1_NORM: l1_norm,
    constants.L2_NORM: l2_norm,
    constants.L_MAX: l_max,
    constants.INTERSECTION: intersection,
    constants.COSINE_SIMILARITY: cosine_similarity
}

def similarity_measure_of_images(fd, image_id_1, image_id_2, similarity_measure):
    image_v1 = feature_descriptors.read_fd_for_image(fd, image_id_1)
    image_v2 = feature_descriptors.read_fd_for_image(fd, image_id_2)
    return similarity_function_map[similarity_measure](image_v1, image_v2)

# fetches a list of k most similar images given query image ID, k, feature description ID and similarity measure function
def k_most_similar_images(query_image_id, k, fd, similarity_measure_func):
    similarity_measure_values = []
    query_vector = feature_descriptors.read_fd_for_image(fd, query_image_id)
    all_images = feature_descriptors.get_all_image_ids()
    for image_id in all_images:
        if image_id != query_image_id:
            image_vector = feature_descriptors.read_fd_for_image(fd, image_id)
            similarity_measure_values.append((image_id, similarity_measure_func(image_vector, query_vector)))
    return [i[0] for i in sorted(similarity_measure_values, key = lambda x: x[1], reverse=True)[:min(len(similarity_measure_values), k)]]

# shows k most similar images given query image ID, k, feature description ID and similarity measure ID
def show_k_most_similar_images(query_image_id, k, fd, similarity_measure):
    similar_images = k_most_similar_images(query_image_id, k, fd, similarity_function_map[similarity_measure])
    for i in similar_images:
        utils.show_image(i)
