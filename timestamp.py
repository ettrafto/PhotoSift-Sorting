import os
from datetime import datetime
from PIL import Image
import numpy as np


def get_timestamp(image_path):
    with Image.open(image_path) as img:
        exif_data = img._getexif()
        if exif_data:
            # Not all images contain EXIF data, so this needs to be checked
            date_time_original = exif_data.get(36867)  # 36867 is the tag for DateTimeOriginal
            if date_time_original:
                return datetime.strptime(date_time_original, '%Y:%m:%d %H:%M:%S')
    return None

def combine_features_and_timestamps(features, image_paths, time_weight):
    # Extract timestamps and convert to seconds
    timestamps = [get_timestamp(path) for path in image_paths]
    min_time = min(ts for ts in timestamps if ts is not None)
    time_deltas = np.array([(ts - min_time).total_seconds() if ts else 0 for ts in timestamps]).reshape(-1, 1)

    # Normalize time deltas to have a similar scale to image features
    # You might need to adjust the normalization method depending on your feature range
    time_deltas_normalized = time_deltas / np.max(time_deltas)
    time_deltas_normalized *= time_weight

    # Combine features and normalized time deltas
    combined_features = np.hstack((features, time_deltas_normalized))

    return combined_features