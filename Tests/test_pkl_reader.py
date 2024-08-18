import pickle

def print_pickle_file_contents(pickle_file_path):
    try:
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type of data: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            print(f"Number of items: {len(data)}")
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print("The pickle file does not contain a dictionary.")
            print(f"Content: {data}")
    except Exception as e:
        print(f"An error occurred while reading the pickle file: {e}")

# Specify the path to your pickle file
pickle_file_path = 'checkpoint.pkl'  # Adjust the path as needed

# Run the function
print_pickle_file_contents(pickle_file_path)
