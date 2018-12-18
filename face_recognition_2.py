import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

Data_path = 'C:/Users/MD MERAJ ALAM/Downloads/download video/Faces/'
only_files = [f for f in listdir(Data_path) if isfile(join(Data_path,f))]
Training_data, Labels = [], []

for i,files in enumerate(only_files):
    image_path = Data_path+only_files[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images, dtype = np.uint8))
    Labels.append(i)
Labels =np.asarray(Labels,dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_Create()

model.train(np.asarray(Training_data), np.asarray(Labels))

print("Model training Complete")