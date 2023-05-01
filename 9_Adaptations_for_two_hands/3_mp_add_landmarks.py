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

mp_yolo_df = pd.read_csv('MPandYolo.csv')

l_names = []
n = 21  # number of coordinate pairs

for i in range(1, n+1):
  l_names.extend([f"{i}x", f"{i}y"])

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
      
    distance1 = math.sqrt((x-x1)**2 + (1080-y1)**2)
    distance2 = math.sqrt((x-x2)**2 + (1080-y2)**2)

    min_distance = min(distance1, distance2) 
    if min_distance > max_distance: 
        max_distance = min_distance 
        furthest_point = (x, 1080)

  for y in range(height+1):
      
    distance1 = math.sqrt((0-x1)**2 + (y-y1)**2)
    distance2 = math.sqrt((0-x2)**2 + (y-y2)**2)

    min_distance = min(distance1, distance2) 
    if min_distance > max_distance: 
      max_distance = min_distance 
      furthest_point = (0, y)

  for y in range(height+1):
      
    distance1 = math.sqrt((1920-x1)**2 + (y-y1)**2)
    distance2 = math.sqrt((1920-x2)**2 + (y-y2)**2)

    min_distance = min(distance1, distance2) 
    if min_distance > max_distance: 
      max_distance = min_distance 
      furthest_point = (1920, y)
  
  return furthest_point

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.25) as hands:
  for index, row in mp_yolo_df.iterrows():
    grasp = row['grasp']
    handedness = row['handedness']
    aux_row = list(row)[:-3]

    # print(index)
    
    path_ini = 'EpicKitchenFrames/useful/'
    name = row['picture_name']
    
    object_center_x = row['object_min_x'] +row['object_width'] // 2
    object_center_y = row['object_min_y'] +row['object_height'] // 2

    # ext = ''

    # if 'power' in grasp: ext = 'power'
    # elif 'precision' in grasp: ext = 'precision'
    # else: ext = 'none'

    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image_path = os.path.join(path_ini, name)
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
    landmarks = []
    # landmarks_left = []
    # landmarks_right = []
    
    for ind, hand_landmarks in enumerate(results.multi_hand_landmarks):
      
      label = results.multi_handedness[ind].classification[0].label
      # print(1,label,handedness)
      if label.lower() != handedness: continue
      # print(2,label)

      # left_hand_exists = row['l_hand_width'] != 0
      # right_hand_exists = row['r_hand_width'] != 0
      
      # There is no object detection in the image
      if row['object_width'] == 0:
        
        # Furthest corner value
        furthest_x = None
        furthest_y = None
        
        # furthest_x_r = None
        # furthest_y_r = None

        # if left_hand_exists and not right_hand_exists:
        #   l_center_x = row['l_hand_min_x'] + row['l_hand_width']//2
        #   l_center_y = row['l_hand_min_y'] + row['l_hand_height']//2
          
        #   # We set the object the furthest away from the hand, AKA the furthest corner
        #   if l_center_x < 1920//2: furthest_x_l = 1920
        #   else : furthest_x_l = 0

        #   if l_center_y < 1080//2: furthest_y_l = 1080
        #   else : furthest_y_l = 0

        #   object_center_x = furthest_x_l
        #   object_center_y = furthest_y_l

        #   aux_row[8] = object_center_x
        #   aux_row[9] = object_center_y

        #   # And set the opposite phantom hand in the opposite corner
        #   aux_row[4] = 1920-furthest_x_l
        #   aux_row[5] = 1080-furthest_y_l
          

        # if right_hand_exists and not left_hand_exists:
        #   r_center_x = row['r_hand_min_x'] + row['r_hand_width']//2
        #   r_center_y = row['r_hand_min_y'] + row['r_hand_height']//2

        #   # We set the object the furthest away from the hand, AKA the furthest corner
        #   if r_center_x < 1920//2: furthest_x_r = 1920
        #   else : furthest_x_r = 0

        #   if r_center_y < 1080//2: furthest_y_r = 1080
        #   else : furthest_y_r = 0

        #   object_center_x = furthest_x_r
        #   object_center_y = furthest_y_r

        #   aux_row[8] = object_center_x
        #   aux_row[9] = object_center_y

        #   # And set the opposite phantom hand in the opposite corner
        #   aux_row[0] = 1920-furthest_x_r
        #   aux_row[1] = 1080-furthest_y_r

        # if left_hand_exists and right_hand_exists:
        center_x = row['hand_min_x'] + row['hand_width']//2
        center_y = row['hand_min_y'] + row['hand_height']//2

        # r_center_x = row['r_hand_min_x'] + row['r_hand_width']//2
        # r_center_y = row['r_hand_min_y'] + row['r_hand_height']//2

        # The furthest point is defined as following:
        # From the 2 distances that appear when a point is selected using the center of the 2 hands,
        # The point returned in this function has the maximum minimum distance between both distances.
        if center_x < 1920//2: furthest_x = 1920
        else : furthest_x = 0

        if center_y < 1080//2: furthest_y = 1080
        else : furthest_y = 0

        
        # furthest_x, furthest_y = furthest_point(l_center_x, l_center_y, r_center_x, r_center_y, image_width, image_height)

        object_center_x = furthest_x
        object_center_y = furthest_y

        aux_row[4] = object_center_x
        aux_row[5] = object_center_y

      # print(hand_landmarks.landmark)
      # print(hand_landmarks.landmark)
      # count = 0
      for ind2, mark in enumerate(hand_landmarks.landmark):
        # count += 1
        # print(count)
        if label.lower() != handedness: continue

        x_val = mark.x * image_width
        y_val = mark.y * image_height

        x_val = int(round(x_val))
        y_val = int(round(y_val))

        # if row['object_width'] != 0: 

        #Distance from landmark to object
        dist_x = object_center_x - x_val 
        dist_y = object_center_y - y_val
        
        # else:
        #   dist_x, dist_y = 0, 0
        # print(1111,landmarks)
        landmarks.extend([dist_x, dist_y])
        # print(2222,landmarks)


    # if landmarks == []: 
    #   print('LM VACIOOOOOOOOOOOOOOOOO')
    #   # if object_center_x < 1920//2: furthest_x = 1920
    #   # else : furthest_x = 0

    #   # if object_center_y < 1080//2: furthest_y = 1080
    #   # else : furthest_y = 0
      
    #   # aux_row[0] = furthest_x
    #   # aux_row[1] = furthest_y

    #   # x_val = aux_row[0]
    #   # y_val = aux_row[1]
      
    #   dist_x = object_center_x - x_val 
    #   dist_y = object_center_y - y_val

    #   landmarks.extend([dist_x, dist_y]*21)

    # elif landmarks_right == []: 
    #   if object_center_x < 1920//2: furthest_x_r = 1920
    #   else : furthest_x_r = 0

    #   if object_center_y < 1080//2: furthest_y_r = 1080
    #   else : furthest_y_r = 0
      
    #   aux_row[4] = furthest_x_r
    #   aux_row[5] = furthest_y_r

    #   x_val = aux_row[5]
    #   y_val = aux_row[6]
      
    #   dist_x = object_center_x - x_val 
    #   dist_y = object_center_y - y_val

    #   landmarks_right.extend([dist_x, dist_y]*21)


    aux_row.extend(landmarks)
    # aux_row.extend(landmarks_right)

    aux_row.extend([row['handedness'], row['picture_name'], row['grasp']])

    # print(aux_row)
    # print(df.columns)
    # dists_list.append(name)
    # if 'power' in name:
    #   dists_list.append('power')
    
    # else: dists_list.append('precision')

    df.loc[len(df)] = aux_row

    print(df)
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
      

  df.to_csv('MPandYOLOandLM_dist2Hands_corrections.csv', index=False)
