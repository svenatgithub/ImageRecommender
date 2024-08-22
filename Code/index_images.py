import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import re
from tqdm import tqdm
import io
import gc
import pickle

from config import CHUNK_SIZE, COMPRESS_QUALITY, MAX_IMAGE_SIZE, CHECKPOINT_PATH, RGB_PATH, HSV_PATH, FINAL_EMBEDDINGS_PATH
from db_api import create_connection, create_tables, insert_rgb_histogram, insert_hsv_histogram, insert_embedding

def compress_image(image, quality=COMPRESS_QUALITY, max_size=(512, 512)):
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    img_io = io.BytesIO()

    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.LANCZOS)

    image.save(img_io, format='JPEG', quality=quality, optimize=True)

    img_io.seek(0)
    return Image.open(img_io)

def get_images(PATH_TO_SSD, CHUNK_SIZE):
    regex = r"^.*\.(jpg|jpeg|png)$"
    image_files = []
    
    for root, dirs, files in os.walk(PATH_TO_SSD):
        for file in files:
            if re.match(regex, file, re.IGNORECASE):
                image_files.append(os.path.join(root, file))  # Ensure full path is saved
    
    total_images = len(image_files)
    total_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(total_chunks):
        yield image_files[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], total_chunks, i + 1

def save_checkpoint(data, CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(CHECKPOINT_PATH):
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def save_histograms_as_dicts(rgb_histograms, hsv_histograms, RGB_PATH, HSV_PATH):
    """Save RGB and HSV histograms as dictionaries in CSV files."""
    print("Saving histograms to CSV files...")
    try:
        rgb_records = []
        hsv_records = []

        # Prepare RGB histogram records
        for record in rgb_histograms:
            record_dict = {
                'image_id': record['image_id'],
                'image_path': record['image_path'],
            }
            record_dict.update({f'bin_{i}': value for i, value in enumerate(record['histogram'])})
            rgb_records.append(record_dict)

        # Prepare HSV histogram records
        for record in hsv_histograms:
            record_dict = {
                'image_id': record['image_id'],
                'image_path': record['image_path'],
            }
            record_dict.update({f'bin_{i}': value for i, value in enumerate(record['histogram'])})
            hsv_records.append(record_dict)

        # Save to CSV
        rgb_df = pd.DataFrame(rgb_records)
        hsv_df = pd.DataFrame(hsv_records)

        rgb_df.to_csv(RGB_PATH, index=False)
        print(f"RGB histograms saved successfully in {RGB_PATH}")

        hsv_df.to_csv(HSV_PATH, index=False)
        print(f"HSV histograms saved successfully in {HSV_PATH}")

    except Exception as e:
        print(f"Failed to save histograms: {e}")

def process_and_save_images(RGB_PATH, HSV_PATH, PATH_TO_SSD, CHUNK_SIZE):
    conn = create_connection('metadata.db')
    create_tables(conn)
    
    if conn is None:
        print("Error! Cannot create the database connection.")
        return
    
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    start_chunk = 0
    if checkpoint:
        start_chunk = checkpoint['chunk_index']
        print(f"Resuming from chunk {start_chunk + 1}")
    
    rgb_histograms = []
    hsv_histograms = []

    for chunk_images, total_chunks, chunk_index in get_images(PATH_TO_SSD, CHUNK_SIZE):
        if chunk_index <= start_chunk:
            continue  # Skip already processed chunks

        gc.collect()  # Clear memory at the start of each chunk
        
        for image_path in tqdm(chunk_images, desc=f'Processing chunk {chunk_index}/{total_chunks}', file=sys.stdout):
            try:
                image_id = os.urandom(8).hex()
                image = Image.open(image_path)
                image = compress_image(image)  # Ensure the image is resized

                # Convert the image to a NumPy array in RGB format
                image_array = np.array(image.convert('RGB'))
                
                # Compute RGB histogram
                rgb_hist = cv2.calcHist([image_array], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256]).flatten()
                rgb_histograms.append({
                    'image_id': image_id,
                    'image_path': image_path,  # Ensure full path is saved
                    'histogram': rgb_hist
                })

                # Convert image to HSV and compute histogram
                hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
                hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256]).flatten()
                hsv_histograms.append({
                    'image_id': image_id,
                    'image_path': image_path,  # Ensure full path is saved
                    'histogram': hsv_hist
                })

                # Insert histograms into the database
                insert_rgb_histogram(conn, image_id, image_path, rgb_hist)
                insert_hsv_histogram(conn, image_id, image_path, hsv_hist)
                
                # Generate and store embedding (assuming some embedding function or model)
                # embedding = your_embedding_function(image)
                # embeddings[image_id] = embedding
                # insert_embedding(conn, image_id, image_path, embedding)

                # Save histograms incrementally to avoid memory issues
                if len(rgb_histograms) >= CHUNK_SIZE:
                    save_histograms_as_dicts(rgb_histograms, hsv_histograms, RGB_PATH, HSV_PATH)
                    rgb_histograms.clear()
                    hsv_histograms.clear()

            except (IOError, OSError) as e:
                print(f"Failed to process image {image_path} due to corruption: {e}")
            except MemoryError:
                print(f"Failed to process image {image_path} due to insufficient memory.")
                gc.collect()  # Attempt to clean up memory immediately
                continue  # Skip to the next image
        
        save_checkpoint({'chunk_index': chunk_index}, CHECKPOINT_PATH)
        gc.collect()  # Clear memory after each chunk

    # Save histograms and embeddings after all chunks are processed
    save_histograms_as_dicts(rgb_histograms, hsv_histograms, RGB_PATH, HSV_PATH)
   
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    conn.close()
    return rgb_histograms, hsv_histograms

if __name__ == '__main__':
    process_and_save_images(RGB_PATH, HSV_PATH, 'PATH_TO_SSD', CHUNK_SIZE)
