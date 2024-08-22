import os
import numpy as np
import pickle
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torch
from PIL import Image
from config import PATH_TO_SSD, FINAL_EMBEDDINGS_PATH, CHECKPOINT_PATH, CHECKPOINT_INTERVAL
import logging

logging.basicConfig(level=logging.INFO)

def load_image(image_path):
    logging.info(f"Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        logging.info(f"Image loaded and preprocessed: shape={image.shape}, dtype={image.dtype}")
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def get_embedding(model, img):
    logging.info("Extracting embedding...")
    try:
        with torch.no_grad():
            embedding = model(img).cpu().numpy().flatten()
        logging.info(f"Embedding extracted: shape={embedding.shape}, dtype={embedding.dtype}")
        return embedding
    except Exception as e:
        logging.error(f"Error extracting embedding: {e}")
        return None

def save_embeddings(embeddings, path):
    logging.info(f"Saving embeddings to {path}")
    try:
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
        logging.info("Embeddings saved successfully")
    except Exception as e:
        logging.error(f"Error saving embeddings: {e}")

def process_images_and_save_embeddings(image_paths, CHUNK_SIZE, CHECKPOINT_PATH, FINAL_EMBEDDINGS_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_pretrained('efficientnet-b7').to(device)
    model.eval()

    start_index = 0
    embeddings = {}

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            checkpoint_data = pickle.load(f)
            start_index = checkpoint_data['start_index']
            embeddings = checkpoint_data['embeddings']
        logging.info(f"Resuming from image index {start_index}")
    else:
        logging.info("Starting from the beginning...")

    logging.info(f"Total images to process: {len(image_paths) - start_index}")

    with tqdm(total=len(image_paths) - start_index, desc="Processing Images") as pbar:
        for i in range(start_index, len(image_paths)):
            image_path = image_paths[i]
            img = load_image(image_path)
            if img is not None:
                img = img.to(device)
                embedding = get_embedding(model, img)
                if embedding is not None:
                    image_id = os.path.basename(image_path)
                    embeddings[image_id] = embedding
                    logging.info(f"Embedding added for image_id: {image_id}")

            pbar.update(1)

            if (i + 1) % CHECKPOINT_INTERVAL == 0 or (i + 1) == len(image_paths):
                with open(CHECKPOINT_PATH, 'wb') as f:
                    pickle.dump({
                        'start_index': i + 1,
                        'embeddings': embeddings,
                    }, f)
                logging.info(f"Checkpoint saved at image index {i + 1}")

    logging.info(f"Total embeddings collected: {len(embeddings)}")
    save_embeddings(embeddings, FINAL_EMBEDDINGS_PATH)

def test_load_pickle(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Pickle file loaded successfully from {path}")
        logging.info(f"Type of loaded data: {type(data)}")
        logging.info(f"Number of items: {len(data)}")
        if len(data) > 0:
            logging.info(f"Sample item: {list(data.items())[0]}")
        else:
            logging.info("No items in the loaded data")
    except Exception as e:
        logging.error(f"Error loading pickle file: {e}")

def get_image_paths(directory):
    supported_extensions = ('.jpg', '.jpeg', '.png')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(supported_extensions)]

if __name__ == "__main__":
    image_paths = get_image_paths(PATH_TO_SSD)
    CHUNK_SIZE = 100

    process_images_and_save_embeddings(image_paths, CHUNK_SIZE, CHECKPOINT_PATH, FINAL_EMBEDDINGS_PATH)
    test_load_pickle(FINAL_EMBEDDINGS_PATH)

