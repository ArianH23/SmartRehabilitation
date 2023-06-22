import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# from google.protobuf.json_format import MessageToDict

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

# For static images:
# image_name = 'precision3025.png'
# IMAGE_FILES = [f'frames/precision/{image_name}']
c = ['hand_min_x','hand_min_y','hand_width','hand_height','handedness','picture_name', 'grasp']
df = pd.DataFrame(columns=c)
i = 0

possible_folders = ['original', 'AFP1920', '256', 'AFP300', 'AFC300']
pos_i = 0

count_ori = 0
count_256 = 0
count_AFP300 = 0
count_AFC300 = 0
count_pad = 0
no_detection = 0

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.25) as hands:
  for path, subdirs, files in os.walk('EpicKitchen/data'):
    for name in files:

      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      i = i + 1
      
      # print(name)
      ori_path = path
      # image_path = os.path.join(path, name)
      # image = cv2.flip(cv2.imread(image_path), 1)
      # # Convert the BGR image to RGB before processing.
      # results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  
      # Print handedness and draw hand landmarks on the image.
      # if results.multi_handedness is not None:
      #   print('Handedness:', (results.multi_handedness[0].classification[0].label))

      # if not results.multi_hand_landmarks:
      have_results = False
      for fold in possible_folders:
        dataset = 'EpicKitchen'

        image_path = os.path.join(dataset, fold, name)
        image = cv2.flip(cv2.imread(image_path), 1)
        annotated_image = cv2.flip(cv2.imread(dataset +'/original/'+name), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
          
          label1 = results.multi_handedness[0].classification[0].label
          label1 = label1.lower()
          if label1 in ori_path:
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

              if label2 in ori_path:
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
       
      print(image.shape)
      image_height, image_width = 1080, 1920
      
      
      bb1 = [0, 0, 0, 0]

      d_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

      for ind, hand_landmarks in enumerate(results.multi_hand_landmarks):
        
        max_x = 0
        max_y = 0 
        min_x = 1920
        min_y = 1920

        label = results.multi_handedness[ind].classification[0].label
        label = label.lower()

        if label not in ori_path: continue

        for mark in hand_landmarks.landmark:
          
          x_val = mark.x
          y_val = mark.y

          if 'AFC' in image_path:
            if x_val * 1080 + 420 > max_x: max_x = x_val * 1080 + 420
            if x_val * 1080 + 420 < min_x: min_x = x_val * 1080 + 420

          else:
            if x_val * image_width > max_x: max_x = x_val * image_width
            if x_val * image_width < min_x: min_x = x_val * image_width

          # Due to the original resolution with padding 
          if 'AFP' in image_path:
            if y_val * 1920 - 420 > max_y: max_y = y_val * 1920 - 420
            if y_val * 1920 - 420 < min_y: min_y = y_val * 1920 - 420

          # elif image.shape[0] == 640:
          #   if y_val * 640 + 220 > max_y: max_y = y_val * 640 + 220
          #   if y_val * 640 + 220 < min_y: min_y = y_val * 640 + 220
          
          else:
            if y_val * image_height > max_y: max_y = y_val * image_height
            if y_val * image_height < min_y: min_y = y_val * image_height
  
        
    
        x_dif = max_x - min_x
        y_dif = max_y - min_y
    
        
        min_x = image_width - min_x
        max_x = image_width - max_x
        
        min_x = int(np.round(min_x))
        max_x = int(np.round(max_x))
        
        min_x , max_x = max_x, min_x

        min_x = max(0, min_x)
        min_y = max(0, min_y)

        max_x = min(1920, max_x)
        max_y = min(1080, max_y)

        width = max_x - min_x
        
        height = int(np.round(max_y)) - int(np.round(min_y))

        bb1 = [min_x, int(np.round(min_y)), width, height, label]

        # if label == 'Left':
        #   bb1 = [min_x, int(np.round(min_y)), width, height, 'left']
        # else: 
        #   bb1 = [min_x, int(np.round(min_y)), width, height, 'right']
        # print(max_x, max_y)
        annotated_image = cv2.flip(annotated_image,1)
        annotated_image = cv2.rectangle(annotated_image, ((min_x), int(np.round(min_y))), (max_x, int(np.round(max_y))), (0,255,255), 3)
        annotated_image = cv2.flip(annotated_image,1)

      if bb1==[0,0,0,0] : continue
      d_list = bb1 + [name]
      # print(d_list)
      if 'power' in path:
        d_list.append('power')
      
      elif 'precision' in path: d_list.append('precision')

      else: d_list.append('none')

      df.loc[len(df)] = d_list
      print(df)
      print('Detections in 256 resolution:', count_256)
      print('Detections in 300AfP resolution:', count_AFP300)
      print('Detections in 300AfC resolution:', count_AFC300)
      print('Detections in original 1920x1080 resolution:', count_ori)
      print('Detections in 1920Afp resolution:', count_pad)
      print('No detections:', no_detection)
      print(fold)


      cv2.imwrite(
          'randomBB/'+ name[:-4] + '.png', cv2.flip(annotated_image, 1))


      if not results.multi_hand_world_landmarks:
        continue


df.to_csv('EK_mp.csv', index=False)
