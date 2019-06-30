# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def preduce_labels(data_file, save_file):
	model = load_model('../model_parameters/0,1-4,vgg,adam,50.h5')
	fingers = os.listdir(data_file)
	fingers.sort()
	count = 0
	with open(save_file, 'a') as f:
		for finger in fingers:
			images = os.listdir(os.path.join(data_file, finger))
			images.sort()
			for image in images:
				count += 1
				img = Image.open(os.path.join(data_file, finger, image))
				w, h = img.size
				img = img.resize((224,224), Image.ANTIALIAS)
				img = img.convert("RGB")
				im_dat = np.array(img, dtype=np.float32)
				pixel_max = np.max(im_dat)
				im_dat = im_dat / pixel_max
				im_data = []
				im_data.append(im_dat)
				im_data = np.array(im_data)
				preds = model.predict(im_data)
				preds[preds < 0] = 0 # 输出的坐标都是归一化的
				preds[preds > 1] = 1
				
				#preds = preds.astype(np.int32)
				f.writelines([image, '\t', str(preds[0][0]), '\t', str(preds[0][1]), '\t', str(preds[0][2]), '\t', str(preds[0][3]), '\r\n'])
            	#f.write('\r\n')
	print(count)

if __name__ == '__main__':
	data_file = '/home/hza/deeplearning_project/FingerVeinRecognition/datasets/data/SDUMLA-HMT/Raw_aug/train_382'
	save_file = '/home/hza/deeplearning_project/FingerVeinRecognition/datasets/data/SDUMLA-HMT/Raw_aug/train_bbox_labels.txt'
	preduce_labels(data_file, save_file)



