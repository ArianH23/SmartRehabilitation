import os
from ultralytics import YOLO
import pandas as pd
import numpy as np
# Load a model
# model = YOLO("yolov8x8.yaml")  # build a new model from scratch
# model = YOLO("yolov8x6.pt")  # load a pretrained model (recommended for training)

mp_df = pd.read_csv('dataset_dist_to_min/mp_bb.csv')

# Get a list of all the files and folders in the directory
path = 'runs/detect'
list_dir = os.listdir(path)
last_folder_number = 0

bb_df = pd.DataFrame(columns=['col{}'.format(i) for i in range(1, 11)])



for index, row in mp_df.iterrows():   
    file_name = row['col5']
    path_ext=''

    aux_row = list(row) + [None,None,None,None]

    # Loop through the list and get the last name of the folder
    if 'power' in file_name: path_ext='power'
    elif 'precision' in file_name: path_ext='precision'
    else: path_ext='none'
    
    image = "frames/" + path_ext + "/" + file_name
    model = YOLO("yolov8x6.pt")

    results = model.predict(image, save_txt=True, classes=[42,43,44,46,49,67,76]) 

    print(os.listdir(path))
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            current_folder_number = os.path.basename(item)[7:]
            
            if current_folder_number == '': continue
            
            current_folder_number = int(current_folder_number)

            if current_folder_number is None or current_folder_number > last_folder_number:
                last_folder_number = current_folder_number

    print(last_folder_number)
    file = ''
    reader = None
    for item in os.listdir(path+'/'+'predict'+str(last_folder_number)+'/labels'):
        file = item
        reader = pd.read_csv(path+'/'+'predict'+str(last_folder_number)+'/labels/' +file, sep=' ', header=None)

    if reader is not None:
        print(reader)
        first = reader.iloc[0]
        min_x = int(np.round(first[1]*1920))
        min_y = int(np.round(first[2]*1080))
        width = int(np.round(first[3]*1920))
        height = int(np.round(first[4]*1080))
        
        aux_row[6] = min_x
        aux_row[7] = min_y
        aux_row[8] = width
        aux_row[9] = height

        my_order = [0,1,2,3,6,7,8,9,4,5]
        aux_row = [aux_row[i] for i in my_order]
        
        print(aux_row)
        bb_df.loc[len(bb_df)] = aux_row
        print(bb_df)