from PIL import Image
from index_images import compress_image
from embeddings import get_embedding




def create_embeddings(image_name):
    
    data = []
    print("Open Image...")
    image = Image.open(image_name)
    print("Done!")
    print("Compress Image...")
    image = compress_image(image)
    print("Done!")
    print("Get RGB data...")
    rgb_hist = cv2.calcHist([np.array(image)], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    data[1] = rgb_hist
    print("Done!")
    print("Get HSV data...")
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    data[2] = hsv_hist
    print("Done!")
    print("Get Embeddings...") 
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.eval()
    embeddings = get_embedding(model,image)
    print("Done!")
    data[0] = embeddings
    return data


















