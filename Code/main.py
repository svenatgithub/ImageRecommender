import os
import pandas as pd
from config import CHECKPOINT_PATH, RGB_PATH, HSV_PATH, PATH_TO_SSD, CHUNK_SIZE, PICKLE_PATH
from index_images import process_and_save_images
from embeddings import process_images_and_save_embeddings
from db_api import create_tables, create_connection, get_image_ids
from create_plot_knn import load_embeddings_from_pickle, load_histograms_from_csv, find_top_5_similar_images, plot_top_5_images


def main():
# Step 1: Process and save RGB and HSV histograms
    print("Processing and saving RGB and HSV histograms...")
    process_and_save_images(RGB_PATH, HSV_PATH, PATH_TO_SSD, CHUNK_SIZE)

# Step 2: Create a connection to the database
    conn = create_connection('metadata.db')
    if conn is None:
        print("Error! Cannot create the database connection.")
        return

# Ensure tables are created
    create_tables(conn)

# Step 3: Process images and save embeddings directly as a pickle file (we first were saving the embeddings as df and then converted but this approah is quite CPU heavy). 
    print("Processing and saving embeddings...")
    image_paths = [os.path.join(PATH_TO_SSD, f) for f in os.listdir(PATH_TO_SSD) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    process_images_and_save_embeddings(image_paths, CHUNK_SIZE, CHECKPOINT_PATH, PICKLE_PATH)

    # Step 4: Retrieve all image_ids from the database
    image_ids = get_image_ids(conn)

    # Step 5: Perform KNN analysis for each image_id
    for image_id in image_ids:
        print(f"Analyzing image_id: {image_id}")
        
        # Step 6: Find and plot top 5 most similar images
        embeddings = load_embeddings_from_pickle(os.path.join(PICKLE_PATH, 'embeddings.pkl'))
        rgb_histograms, hsv_histograms = load_histograms_from_csv(RGB_PATH, HSV_PATH)
        
        image_index = image_ids.index(image_id)
        top_5_indices = find_top_5_similar_images(embeddings, rgb_histograms, hsv_histograms, image_index)
        plot_top_5_images(top_5_indices, PATH_TO_SSD, image_paths)

    print("Processing, KNN creation, and plotting complete.")

    # Step 7: Close the database connection
    conn.close()

if __name__ == "__main__":
    main()

