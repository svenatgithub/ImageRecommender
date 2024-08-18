import os
import pickle
import numpy as np
from tqdm import tqdm
from config import CHUNK_SIZE, PICKLE_PATH
from db_api import insert_embedding

def process_images_and_save_embeddings(image_paths, chunk_size, checkpoint_path):
    embeddings = {}
    start_index = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            start_index = checkpoint['index']
            embeddings = checkpoint.get('embeddings', {})
            print(f"Resuming from index {start_index}, loaded {len(embeddings)} embeddings from checkpoint.")
    else:
        print("Starting from scratch...")

    for i in tqdm(range(start_index, len(image_paths), chunk_size)):
        chunk = image_paths[i:i + chunk_size]
        for image_path in chunk:
            image_id = os.path.basename(image_path).split('.')[0]  # Use file name without extension as image_id
            # Generate embedding (placeholder, replace with actual embedding generation logic)
            embedding = np.random.rand(128)  # Example: 128-dimension random embedding
            embeddings[image_id] = embedding

      # Insert into database
            insert_embedding(image_id, image_path, embedding)

      # Save progress to checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'index': i + chunk_size, 'embeddings': embeddings}, f)

    print(f"Finished processing images. Total embeddings generated: {len(embeddings)}")

  # Save the final embeddings dictionary (here please dont get confused wwhen trying to read the content of embeddings.pkl with test_pkl_reader.pkl because 
  # it is first saved as checkpoint.pkl and only concatenated and replaced with embeddings.pkl after main.py was 100% executed.
    save_embeddings_as_pickle(embeddings)

    return embeddings, list(embeddings.keys())

def save_embeddings_as_pickle(embeddings):
# Saving the embeddings dictionary to a pickle file
    try:
        with open(os.path.join(PICKLE_PATH, 'embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved successfully to {os.path.join(PICKLE_PATH, 'embeddings.pkl')}")
    except Exception as e:
        print(f"Error saving embeddings to pickle file: {e}")
