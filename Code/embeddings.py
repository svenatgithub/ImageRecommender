import os
import numpy as np
import pickle
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torch
from PIL import Image
from config import PICKLE_PATH, CHECKPOINT_PATH, CHECKPOINT_INTERVAL

#EfficientNet b0 only can work with images of size 224x224 so we need to adjust the image size and dimensions
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
    return image

# Function to extract embeddings using EfficientNet
def get_embedding(model, img):
    with torch.no_grad():
        embedding = model(img).numpy().flatten()
    return embedding

# Function to save embeddings as a dictionary in a pickle file
def save_embeddings_as_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Embeddings saved to {path}")

# Function to process images and save checkpoints
def process_images_and_save_embeddings(image_paths, checkpoint_interval, checkpoint_path, pickle_path):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.eval()

    start_index = 0
    embeddings_data = []
    
# Load checkpoints if they exist
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            start_index = checkpoint_data['start_index']
            embeddings_data = checkpoint_data['embeddings_data']
        print(f"Resuming from image index {start_index}")
    else:
        print("Starting from the beginning...")

    with tqdm(total=len(image_paths) - start_index, desc="Processing Images") as pbar:
        for i in range(start_index, len(image_paths)):
            image_path = image_paths[i]
            img = load_image(image_path)
            embedding = get_embedding(model, img)
            image_id = os.path.basename(image_path)
            
            # Append the data as a dictionary
            embeddings_data.append({
                'image_id': image_id,
                'image_path': image_path,
                'embedding': embedding
            })
            
            pbar.update(1)
            
            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(image_paths):
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'start_index': i + 1,
                        'embeddings_data': embeddings_data,
                    }, f)
                print(f"Checkpoint saved at image index {i + 1}")
    
    # Save final embeddings as a pickle file
    save_embeddings_as_pickle(embeddings_data, pickle_path)
