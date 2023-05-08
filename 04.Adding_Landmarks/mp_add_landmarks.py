import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

mp_yolo_df = pd.read_csv('data/MPandYOLO_dist.csv')

l_names = []
n = 21  # number of coordinate pairs

for i in range(1, n+1):
    l_names.extend([f"l{i}x", f"l{i}y"])

df = pd.DataFrame(columns=list(mp_yolo_df.columns[:-2]) + l_names + ['picture_name', 'grasp'])
i = 0

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.25) as hands:
  for index, row in mp_yolo_df.iterrows():
    
    aux_row = list(row)[:-2]

    print(index)
    
    path_ini = 'frames'
    name = row['picture_name']
    
    object_center_x = row['object_min_x'] +row['object_width']//2
    object_center_y = row['object_min_y'] +row['object_height']//2

    ext = ''

    if 'power' in name: ext = 'power'
    elif 'precision' in name: ext = 'precision'
    else: ext = 'none'

    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image_path = os.path.join(path_ini, ext, name)
    image = cv2.flip(cv2.imread(image_path), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    #   print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()

    max_x = 0
    max_y = 0 
    min_x = 1920
    min_y = 1920

    landmarks_values = []
    
    for hand_landmarks in results.multi_hand_landmarks:
      for mark in hand_landmarks.landmark:

        x_val = mark.x * image_width
        y_val = mark.y * image_height

        x_val = int(round(x_val))
        y_val = int(round(y_val))

        if row['object_width'] != 0: 

          dist_x = x_val - object_center_x
          dist_y = y_val - object_center_y
        
        else:
           dist_x, dist_y = 0, 0
           
        landmarks_values.extend([dist_x, dist_y])

    aux_row.extend(landmarks_values)
    aux_row.extend([row['picture_name'], row['grasp']])
    # print(aux_row)
    # print(aux_row)
    # dists_list.append(name)
    # if 'power' in name:
    #   dists_list.append('power')
    
    # else: dists_list.append('precision')


    df.loc[len(df)] = aux_row


    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     hand_landmarks,
    #     mp_hands.HAND_CONNECTIONS,
    #     mp_drawing_styles.get_default_hand_landmarks_style(),
    #     mp_drawing_styles.get_default_hand_connections_style())

    # annotated_image = cv2.rectangle(annotated_image, (int(np.round(min_x)), int(np.round(min_y))), (int(np.round(max_x)), int(np.round(max_y))), (255,255,0), 3)
#   print(len(hc_x_norm), len(res_x))
#   for i in range(len(res_x)):

#     annotated_image = cv2.putText(
#      annotated_image, #numpy array on which text is written
#      f'({round(hc_x_norm[i],2)}, {round(hc_y_norm[i],2)})', #text
#      (int(np.round(res_x[i])), (int(np.round(res_y[i])))), #position at which writing has to start
#      cv2.FONT_HERSHEY_SIMPLEX, #font family
#      0.3, #font size
#      (255, 255, 255, 255), #font color
#      1) #font stroke

    # cv2.imwrite(
    #     'mediapipe_photos/'+ name[:-4] + '.png', cv2.flip(annotated_image, 1))
    # cv2.imwrite(
    #     'mediapipe_photos/'+ name[:-4] + 'rev.png', annotated_image)
    # Draw hand world landmarks.
    # if not results.multi_hand_world_landmarks:
    #   continue
    #   for hand_world_landmarks in results.multi_hand_world_landmarks:
    #     mp_drawing.plot_landmarks(
    #       hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
      

df.to_csv('dataset_dist_to_min/MPandYOLOandLM_dist.csv', index=False)
