import os
from ultralytics import YOLO
import pandas as pd
import numpy as np
# Load a model
# model = YOLO("yolov8x8.yaml")  # build a new model from scratch
# model = YOLO("yolov8x6.pt")  # load a pretrained model (recommended for training)

mp_df = pd.read_csv('data/mp_bb.csv')

# Get a list of all the files and folders in the directory
path = 'runs/detect'
list_dir = os.listdir(path)
last_folder_number = 0

bb_df = pd.DataFrame(columns=['hand_min_x','hand_min_y', 'hand_width', 'hand_length',
                              'object_min_x','object_min_y', 'object_width', 'object_height'
                              'picture_name', 'grasp'])


model = YOLO("yolov8x6.pt")


for index, row in mp_df.iterrows():   
    file_name = row['col5']
    path_ext=''

    aux_row = list(row) + [0,0,0,0]

    # Loop through the list and get the last name of the folder
    if 'power' in file_name: path_ext='power'
    elif 'precision' in file_name: path_ext='precision'
    else: path_ext='none'
    
    image = "frames/" + path_ext + "/" + file_name

    results = model.predict(image, save_txt=True, classes=[42,43,44,46,49,64,67,76]) 

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

    closest_distance = 1920*1080

    closest_min_x = 0
    closest_min_y = 0
    closest_width = 0
    closest_height = 0

    hand_min_x = row[0]
    hand_min_y = row[1]

    if reader is not None:
        print(reader)
        for index2, row2 in reader.iterrows():

            min_x = int(np.round(row2[1]*1920))
            min_y = int(np.round(row2[2]*1080))

            distance = np.sqrt((closest_min_x-min_x)^2 + (closest_min_y-min_y)^2)
            
            if distance < closest_distance:
                closest_distance = distance
                width = int(np.round(row2[3]*1920))
                height = int(np.round(row2[4]*1080))
                
                aux_row[6] = min_x
                aux_row[7] = min_y
                aux_row[8] = width
                aux_row[9] = height

        my_order = [0,1,2,3,6,7,8,9,4,5]
        aux_row = [aux_row[i] for i in my_order]
            
        print(aux_row)
        bb_df.loc[len(bb_df)] = aux_row
        print(bb_df)
        os.remove(path+'/'+'predict'+str(last_folder_number)+'/labels/' +file)

    else:
        my_order = [0,1,2,3,6,7,8,9,4,5]
        aux_row = [aux_row[i] for i in my_order]
        print(aux_row)
        bb_df.loc[len(bb_df)] = aux_row
        print(bb_df)

bb_df.to_csv('data/MPandYOLO.csv', index = False)