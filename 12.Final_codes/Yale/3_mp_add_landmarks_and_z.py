import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math

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

mp_yolo_df = pd.read_csv('yale_mp_yolo.csv')
# data_to_replace = pd.read_csv('dataset_dist_to_min/mp_bb_salad.csv')

# data_to_replace = data_to_replace.iloc[:, :8]
# print(data_to_replace)

# # Just doing a replace of values to avoid running the YOLO algorithm again
# for i in range(8):
#   mp_yolo_df.iloc[:, i] = data_to_replace.iloc[:,i]


l_names = []
n = 21  # number of coordinate pairs

for i in range(1, n+1):
  l_names.extend([f"{i}x", f"{i}y", f"{i}z"])


df = pd.DataFrame(columns=list(mp_yolo_df.columns[:-3]) + l_names + ['handedness','picture_name', 'grasp'])
i = 0

def furthest_point(x1, y1, x2, y2, width, height):
  max_distance = 0 
  furthest_point = None 
  for x in range(width+1):
    
    distance1 = math.sqrt((x-x1)**2 + (0-y1)**2)
    distance2 = math.sqrt((x-x2)**2 + (0-y2)**2)

    min_distance = min(distance1, distance2) 
  
  if min_distance > max_distance: 
    max_distance = min_distance 
    furthest_point = (x, 0)

  for x in range(width+1):
      
    distance1 = math.sqrt((x-x1)**2 + (height-y1)**2)
    distance2 = math.sqrt((x-x2)**2 + (height-y2)**2)

    min_distance = min(distance1, distance2) 
    if min_distance > max_distance: 
        max_distance = min_distance 
        furthest_point = (x, height)

  for y in range(height+1):
      
    distance1 = math.sqrt((0-x1)**2 + (y-y1)**2)
    distance2 = math.sqrt((0-x2)**2 + (y-y2)**2)

    min_distance = min(distance1, distance2) 
    if min_distance > max_distance: 
      max_distance = min_distance 
      furthest_point = (0, y)

  for y in range(height+1):
      
    distance1 = math.sqrt((width-x1)**2 + (y-y1)**2)
    distance2 = math.sqrt((width-x2)**2 + (y-y2)**2)

    min_distance = min(distance1, distance2) 
    if min_distance > max_distance: 
      max_distance = min_distance 
      furthest_point = (width, y)
  
  return furthest_point

possible_folders = ['']

count_ori = 0
count_256 = 0
count_AFP300 = 0
count_AFC300 = 0
count_pad = 0
no_detection = 0

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.15) as hands:
  for index, row in mp_yolo_df.iterrows():
    print(index+1,'/',mp_yolo_df.shape[0])
    grasp = row['grasp']
    handedness = row['handedness']
    aux_row = list(row)[:-3]

    # print(index)
    
    path_ini = 'EpicKitchenFrames/useful/'
    name = row['picture_name']
    
    object_center_x = row['object_min_x'] +row['object_width'] // 2
    object_center_y = row['object_min_y'] +row['object_height'] // 2
    
    annotated_image = cv2.imread(name)
    
    have_results = False

    for fold in possible_folders:
      image_path = os.path.join(name)
      image = cv2.flip(cv2.imread(image_path), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print handedness and draw hand landmarks on the image.
      #   print('Handedness:', results.multi_handedness)
      if results.multi_hand_landmarks:
          
        label1 = results.multi_handedness[0].classification[0].label
        label1 = label1.lower()
        if label1 in handedness:
          have_results = True
          #['256', 'AFP300', 'AFC300', 'original', 'AFC640', 'padded']
          if fold == '256': count_256 += 1
          elif fold == 'AFP300': count_AFP300 += 1
          elif fold == 'AFC300': count_AFC300 += 1
          elif fold == 'original': count_ori += 1
          else: count_pad += 1
          break
        
        else:
          if len(results.multi_handedness) > 1:
            label2 = results.multi_handedness[1].classification[0].label
            label2 = label2.lower()

            if label2 in handedness:
              have_results = True
              if fold == '256': count_256 += 1
              elif fold == 'AFP300': count_AFP300 += 1
              elif fold == 'AFC300': count_AFC300 += 1
              elif fold == 'original': count_ori += 1
              else: count_pad += 1
              break
      
    if not have_results:
      no_detection += 1
      continue

    max_x = 0
    max_y = 0 
    min_x = 640
    min_y = 640

    landmarks = []
    
    rDect = False
    for ind, hand_landmarks in enumerate(results.multi_hand_landmarks):
      if rDect: continue
      label = results.multi_handedness[ind].classification[0].label

      if label.lower() != handedness: continue
      if label =='left': continue
      else: rDect = True
      # left_hand_exists = row['l_hand_width'] != 0
      # right_hand_exists = row['r_hand_width'] != 0
      
      # There is no object detection in the image
      if row['object_width'] == 0:
        
        center_x = row['hand_min_x'] + row['hand_width']//2
        center_y = row['hand_min_y'] + row['hand_height']//2

        # r_center_x = row['r_hand_min_x'] + row['r_hand_width']//2
        # r_center_y = row['r_hand_min_y'] + row['r_hand_height']//2

        # The furthest point is defined as following:
        # From the 2 distances that appear when a point is selected using the center of the 2 hands,
        # The point returned in this function has the maximum minimum distance between both distances.
        if center_x < 640//2: furthest_x = 640
        else : furthest_x = 0

        if center_y < 480//2: furthest_y = 480
        else : furthest_y = 0

        
        # furthest_x, furthest_y = furthest_point(l_center_x, l_center_y, r_center_x, r_center_y, image_width, image_height)

        object_center_x = furthest_x
        object_center_y = furthest_y

        aux_row[4] = object_center_x
        aux_row[5] = object_center_y

      
      for mark in hand_landmarks.landmark:

        z_val = mark.z

        if 'AFC' in image_path:
          x_val = mark.x * 480 + 420
          
        else:
          x_val = mark.x * 640
        # Because the image is flipped
        x_val = 640 - x_val

      # Due to the original resolution with padding 
        if 'AFP' in image_path:
          y_val = mark.y * 640 - 420
        else:
          y_val = mark.y * 480
        # print(mark.x, x_val, fold)
        x_val = int(round(x_val))
        y_val = int(round(y_val))

        # if row['object_width'] != 0: 

        #Distance from landmark to object
        dist_x = object_center_x - x_val 
        dist_y = object_center_y - y_val
        
        # else:
        #   dist_x, dist_y = 0, 0
        
        landmarks.extend([dist_x, dist_y, z_val])


     
      aux_row.extend(landmarks)
      # aux_row.extend(landmarks_right)

      aux_row.extend([row['handedness'], row['picture_name'], row['grasp']])

      if len(aux_row) == 74:
        df.loc[len(df)] = aux_row
    
    
    print(df)
    annotated_image = cv2.rectangle(annotated_image, (row['hand_min_x'], row['hand_min_y']), (row['hand_min_x'] + row['hand_width'], row['hand_min_y']+ row['hand_height']), (0, 255, 255), 3)
    annotated_image = cv2.rectangle(annotated_image, (row['object_min_x'], row['object_min_y']), (row['object_min_x'] +row['object_width'], row['object_min_y'] +row['object_height']), (255,255,0), 3)
    for i in range(1,22):
      if (i-1)%4==0 or i==1:
        annotated_image = cv2.line(annotated_image, (object_center_x-df.loc[len(df)-1, f'{i}x'], object_center_y-df.loc[len(df)-1, f'{i}y']), (object_center_x, object_center_y), (0,255,0), 2)

    cv2.imwrite('randomLM/'+ name[:-4] + '.png', annotated_image)

    print('Detections in 256 resolution:', count_256)
    print('Detections in 300AfP resolution:', count_AFP300)
    print('Detections in 300AfC resolution:', count_AFC300)
    print('Detections in original 1920x1080 resolution:', count_ori)
    print('Detections in 1920Afp resolution:', count_pad)
    print('No detections:', no_detection)

  df.to_csv('yale_mp_yolo_mp.csv', index=False)
