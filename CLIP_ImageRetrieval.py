import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import chromadb
from tqdm import tqdm


ROOT = 'path'
CLASS_NAME = sorted(os.listdir(ROOT))

embedding_function = OpenCLIPEmbeddingFunction()

def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + '/' + filename
            files_path.append(filepath)
            
    return files_path

data_path = f"{ROOT}"
files_path = get_files_path(path=data_path)



def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)

def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath).convert('RGB').resize((224, 224))
        image = np.array(image)
        embedding = get_single_image_embedding(image=image)
        embeddings.append(embedding)
    
    collection.add(embeddings=embeddings, ids=ids)
    
    
# Create a Chroma Client
chroma_client = chromadb.Client()
# Create a collection
l2_collection = chroma_client.get_or_create_collection(name="l2_collection", metadata={"hnsw:space": "l2"}) 
cosine_collection = chroma_client.get_or_create_collection(name="Cosine_collection", metadata={"hnsw:space": "cosine"})

add_embedding(collection=l2_collection, files_path=files_path)


def search(image_path, collection, n_results):
    query_image = Image.open(image_path).convert('RGB').resize((224, 224))
    query_image = np.array(query_image)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results



def plot_results(query_path, files_path, results):
    ids_list = results['ids'][0]
    print(ids_list)
    ids_list = [int(id_str.split('_')[1]) for id_str in ids_list]
    query = Image.open(query_path)
    query = query.resize((224, 224))
    plt.figure(figsize=(10, 10))
    plt.subplot(4, 4, 1)
    plt.imshow(query)
    plt.title("Query")
    
    plot_idx = 2
    for ids, path in enumerate(files_path):
        if ids in ids_list:
            img = Image.open(path)
            img = img.resize((224, 224))
            plt.subplot(4, 4, plot_idx)
            plt.imshow(img)
            plt.title(f"Top {ids}")
            plot_idx += 1
    plt.show()
    

test_path = f"{ROOT}"
test_files_path = get_files_path(path=test_path)
test_path = test_files_path[1]
l2_results = search(image_path=test_path, collection=l2_collection, n_results=5)
plot_results(query_path=test_path, files_path=files_path, results=l2_results)