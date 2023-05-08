import os
from ultralytics import YOLO
import pandas as pd
import numpy as np
# Load a model
# model = YOLO("yolov8x8.yaml")  # build a new model from scratch
# model = YOLO("yolov8x6.pt")  # load a pretrained model (recommended for training)

mp_df = pd.read_csv('dataset_dist_to_min/mp_bb_salad.csv')

# Get a list of all the files and folders in the directory
path = 'runs/detect'
list_dir = os.listdir(path)
last_folder_number = 0

c = ['l_hand_min_x','l_hand_min_y','l_hand_width','l_hand_height',
    'r_hand_min_x','r_hand_min_y','r_hand_width','r_hand_height',
    'object_min_x','object_min_y', 'object_width', 'object_height', 
    'picture_name', 'grasp']

bb_df = pd.DataFrame(columns=c)


model = YOLO("yolov8x6.pt")


for index, row in mp_df.iterrows():   
    file_name = row['picture_name']
    grasp = row['grasp']
    path_ext=''

    aux_row = list(row) + [0,0,0,0]
    print(aux_row)
    # Loop through the list and get the last name of the folder
    if 'power' in grasp: path_ext='power'
    elif 'precision' in grasp: path_ext='precision'
    else: path_ext='none'
    
    image = "frames_salad/" + path_ext + "/" + file_name

    results = model.predict(image, save_txt=True, classes=[34, 39, 41, 42, 43, 44, 45, 46, 49, 50, 76]) 

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


    closest_min_x = 0
    closest_min_y = 0
    closest_width = 0
    closest_height = 0

    l_hand_min_x = row[0]
    l_hand_min_y = row[1]
    l_hand_max_x = l_hand_min_x + row[2]
    l_hand_max_x = l_hand_min_x + row[3]

    l_hand_center_x = row[0] + row[2] // 2
    l_hand_center_y = row[1] + row[3] // 2


    r_hand_min_x = row[4]
    r_hand_min_y = row[5]
    r_hand_max_x = r_hand_min_x + row[6]
    r_hand_max_x = r_hand_min_x + row[7]

    r_hand_center_x = row[4] + row[6] // 2
    r_hand_center_y = row[5] + row[7] // 2

    best_iou = 5000
    closest_distance = 1920*1080

    if reader is not None:
        print(reader)
        for index2, row2 in reader.iterrows():

            min_x = int(np.round(row2[1]*1920))
            min_y = int(np.round(row2[2]*1080))
            max_x = min_x + int(np.round(row2[3]*1920))
            max_y = min_y + int(np.round(row2[4]*1080))

            center_x = min_x + int(np.round(row2[3]*1920)) // 2
            center_y = min_y + int(np.round(row2[4]*1080)) // 2

            lxA = max(l_hand_min_x, min_x)
            lyA = max(l_hand_min_y, min_y)
            lxB = min(l_hand_max_x, max_x)
            lyB = min(l_hand_max_x, max_y)

            rxA = max(r_hand_min_x, min_x)
            ryA = max(r_hand_min_y, min_y)
            rxB = min(r_hand_max_x, max_x)
            ryB = min(r_hand_max_x, max_y)

            interAreaL = max(0, lxB - lxA + 1) * max(0, lyB - lyA + 1)
            interAreaR = max(0, rxB - rxA + 1) * max(0, ryB - ryA + 1)

            distanceL = np.sqrt((center_x - l_hand_center_x)^2 + (center_y - l_hand_center_y)^2)
            distanceR = np.sqrt((center_x - r_hand_center_x)^2 + (center_y - r_hand_center_y)^2)
            
            if interAreaL > best_iou:
                best_iou = interAreaL
                width = int(np.round(row2[3]*1920))
                height = int(np.round(row2[4]*1080))
                
                aux_row[10] = min_x
                aux_row[11] = min_y
                aux_row[12] = width
                aux_row[13] = height

            if interAreaR > best_iou:
                best_iou = interAreaR
                width = int(np.round(row2[3]*1920))
                height = int(np.round(row2[4]*1080))
                
                aux_row[10] = min_x
                aux_row[11] = min_y
                aux_row[12] = width
                aux_row[13] = height

            if distanceL < closest_distance:
                closest_distance = distanceL
                width = int(np.round(row2[3]*1920))
                height = int(np.round(row2[4]*1080))

                closest_min_x = min_x
                closest_min_y = min_y
                closest_width = width
                closest_height = height

            if distanceR < closest_distance:
                closest_distance = distanceR
                width = int(np.round(row2[3]*1920))
                height = int(np.round(row2[4]*1080))

                closest_min_x = min_x
                closest_min_y = min_y
                closest_width = width
                closest_height = height
        # If there's no good intersection, we use the distances 
        # from the object to each hand to decide which object to select.
        print(best_iou)
        if best_iou == 5000:
            aux_row[10] = closest_min_x
            aux_row[11] = closest_min_y
            aux_row[12] = closest_width
            aux_row[13] = closest_height    

        my_order = [0,1,2,3,4,5,6,7,10,11,12,13,8,9]
        aux_row = [aux_row[i] for i in my_order]
            
        bb_df.loc[len(bb_df)] = aux_row
        print(bb_df)
        os.remove(path+'/'+'predict'+str(last_folder_number)+'/labels/' +file)

    else:
        my_order = [0,1,2,3,4,5,6,7,10,11,12,13,8,9]
        aux_row = [aux_row[i] for i in my_order]
        print(aux_row)
        bb_df.loc[len(bb_df)] = aux_row
        print(bb_df)


bb_df.to_csv('dataset_dist_to_min/MPandYOLO2Hands.csv', index = False)