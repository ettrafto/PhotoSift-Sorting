import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

import silhouette
import timestamp



# Load and preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_array_expanded)

# Extract features from an image using ResNet50
def extract_features(image_array, model):
    return model.predict(image_array)

# Get all image paths from the folder
def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

# Main function to process images and group them by similarity 
# With silhoutte method of predetermining n_clusters

def process_images(folder_path, max_k):
    # Load the pre-trained ResNet50 model once
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Get all image paths
    image_paths = get_image_paths(folder_path)

    # Check if there are any images to process
    if not image_paths:
        print("No images found in the specified folder.")
        return []

    # Preprocess and extract features from images
    features = []
    for image_path in image_paths:
        preprocessed_image = preprocess_image(image_path)
        features.append(extract_features(preprocessed_image, model))

    # Convert list of arrays into a 2D array for clustering
    features_array = np.vstack(features)

    # Combining Features array with timestamp data
    combined_features = timestamp.combine_features_and_timestamps(features_array, image_paths, time_weight=10)

    # Find the optimal number of clusters using Silhouette Method
    n_clusters = silhouette.find_optimal_clusters_silhouette(combined_features, max_k)

    # Cluster images using K-means with the optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(combined_features)
    labels = kmeans.labels_

    # Define the output directory for clustered images
    output_folder_path = os.path.join('outImages')
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Create and move images to cluster directories within the output folder
    for cluster_id in range(n_clusters):
        cluster_folder_path = os.path.join(output_folder_path, f'cluster_{cluster_id}')
        os.makedirs(cluster_folder_path, exist_ok=True)

    for label, image_path in zip(labels, image_paths):
        # Use the output folder path for clusters
        destination_path = os.path.join(output_folder_path, f'cluster_{label}', os.path.basename(image_path))
        shutil.move(image_path, destination_path)

    return labels

if __name__ == "__main__":
    folder_path = './images' 
    output = './outImages'
    labels = process_images(folder_path, 30)
    print(labels)