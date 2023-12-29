import numpy as np
import os
import tensorflow as tf
import pydicom

def load_scan(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    slices = [pydicom.read_file(path)]
    return slices


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.int16(image)

def read_image(image_path):
    image_path = image_path.numpy().decode('utf-8')
    full_pixels = get_pixels_hu(load_scan(image_path))
    
    MIN_B= -1024.0
    MAX_B= 3072.0
    data = (full_pixels - MIN_B) / (MAX_B - MIN_B)
    
    return np.squeeze(np.expand_dims(data, axis=-1), axis=0)


class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, stride, name):
        super(PatchExtractor, self).__init__(name=name)
        self.patch_size = patch_size
        self.stride = stride

    def call(self, images):
        patch_depth = tf.shape(images)[-1]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, self.patch_size, self.patch_size, patch_depth])
        return patches

def load_and_preprocess_image(file_path, patch_extractor=None):
    image_data = tf.py_function(read_image, [file_path], tf.float32)
        
    if patch_extractor:
        patches = patch_extractor(image_data)
        return patches

    return image_data

def _load_image_paths(low_dose_ct_training_dataset_dir, load_limited_images, num_images_to_load):
    noisy_image_paths = []
    for root, folder_name, file_names in os.walk(low_dose_ct_training_dataset_dir):
        for file_name in file_names:
            if "QD" in file_name:
                file_path = os.path.join(root, file_name)
                noisy_image_paths.append(file_path)

    clean_image_paths = []
    for root, folder_name, file_names in os.walk(low_dose_ct_training_dataset_dir):
        for file_name in file_names:
            if "FD" in file_name:   
                file_path = os.path.join(root, file_name)
                clean_image_paths.append(file_path)

    noisy_image_paths.sort()
    clean_image_paths.sort()

    if load_limited_images:
        noisy_image_paths = noisy_image_paths[:num_images_to_load]
        clean_image_paths = clean_image_paths[:num_images_to_load]
    
    return noisy_image_paths, clean_image_paths

def _create_image_dataset(image_paths, patch_extractor):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    image_dataset = image_dataset.map(lambda file_path: load_and_preprocess_image(file_path, patch_extractor), num_parallel_calls=tf.data.AUTOTUNE)

    return image_dataset

def load_training_tf_dataset(low_dose_ct_training_dataset_dir='../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_as_patches=False, load_limited_images=False, num_images_to_load=10):
    noisy_image_paths, clean_image_paths = _load_image_paths(low_dose_ct_training_dataset_dir, load_limited_images, num_images_to_load)

    patch_extractor = None
    if load_as_patches:
        patch_extractor = PatchExtractor(patch_size=64, stride=64, name="patch_extractor")

    noisy_image_dataset = _create_image_dataset(noisy_image_paths, patch_extractor)
    clean_image_dataset = _create_image_dataset(clean_image_paths, patch_extractor)

    training_dataset = tf.data.Dataset.zip((noisy_image_dataset, clean_image_dataset))
    training_dataset = training_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    print('loaded dataset') 
    
    # Shuffle entire dataset.
    training_dataset = training_dataset.shuffle(100)

    return training_dataset
