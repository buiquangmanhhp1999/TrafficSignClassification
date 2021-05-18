import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

class TrafficSignDataset(Dataset):
    def __init__(self, image_dir, label_file, target_shape=(32, 32)):
        self.image_dir = image_dir
        self.target_height =target_shape[0]
        self.target_width = target_shape[1]

        # read data 
        self.label_maps, self.images_path, self.labels, self.nSample = self.read_label_data(label_file)


    def read_label_data(self, label_file):
        # load labels data
        class_labels = dict()
        images_path = []
        labels = []

        # read label data from csv file
        label_data = pd.read_csv(label_file)
        number_data = 0

        for dir in os.listdir(self.image_dir):
            images_list = os.listdir(self.image_dir + dir)
            number_data += len(images_list)
            class_name = label_data[label_data.ClassId == int(dir)].values[0][1]
            class_labels[int(dir)] = class_name
            pbar = tqdm(range(len(images_list)), desc='Create data for class {}'.format(str(class_name)),ncols = 100, position=0, leave=True) 
            
            for i in pbar:
                img_name = images_list[i]
                image_path = self.image_dir + dir + '/' + img_name
                images_path.append(image_path)
                labels.append(int(dir))

        
        return class_labels, images_path, labels, number_data

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.target_width, self.target_height), cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)
        img = img / 255.0

        return img

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.labels[idx]
        img = self.read_image(img_path)

        return (img, label)

    def __len__(self):
        return self.nSample

    def visualize_random_images(self, nb_row=2, nb_col=3):
        fig, axes = plt.subplots(nb_row,nb_col, figsize=(18, 18))

        for i,ax in enumerate(axes.flat):
            r = np.random.randint(self.nSample)
            img = cv2.imread(self.images_path[r])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.grid(False)
            ax.axis('off')
            ax.set_title('Label: '+ str(self.labels[r]))


class Collator(object):
    def __call__(self, batch):
        images = []
        labels = []

        for sample in batch:
            img, label = sample
            
            if img is None:
                continue
            images.append(img)
            labels.append(label)

        return torch.FloatTensor(images), torch.LongTensor(labels)