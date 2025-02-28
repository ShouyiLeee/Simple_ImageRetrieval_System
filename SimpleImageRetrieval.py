import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = 'C:\\Users\\Acer\OneDrive\\Desktop\\NCKH\\Dataset\\ViVQA4Edu'
CLASS_NAME = sorted(os.listdir(ROOT))


def read_image_from_path(path, size=(224, 224)):
    img = Image.open(path).convert('RGB').resize(size)
    return np.array(img)

def folder_to_images(folder, size=(224, 224)):
    list_dir = [folder + '/' + filename for filename in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, name in enumerate(list_dir):
        images_np[i] = read_image_from_path(name, size)
        images_path.append(name)
    return images_np, images_path

def L1_scores(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


def get_L1_score(root_img_path, query_path, size=(224, 224)):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            rates = L1_scores(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def plot_results(query_path, ls_path_score, reverse=False):
    query = Image.open(query_path)
    query = query.resize((224, 224))
    plt.figure(figsize=(10, 10))
    plt.subplot(4, 4, 1)
    plt.imshow(query)
    plt.title("Query")
    for i, (path, score) in enumerate(ls_path_score):
        plt.subplot(4, 4, i + 2)
        img = Image.open(path)
        img = img.resize((224, 224))
        plt.imshow(img)
        plt.title(f"Top {i}")
        if i>=9:
            break
    plt.show()
    
root_img_path = f"{ROOT}/"
query_path = "C:\\Users\\Acer\\OneDrive\\Desktop\\NCKH\\Dataset\\VIVQA4Edu-2\\Screenshot_618.png"
size = (224, 224)
query, ls_path_score = get_L1_score(root_img_path, query_path, size)
plot_results(query_path, ls_path_score, reverse=False)