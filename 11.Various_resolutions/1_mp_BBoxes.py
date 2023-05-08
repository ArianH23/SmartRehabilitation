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

possible_folders = ['256', 'AFP300', 'AFC300', 'original', 'AFP640', 'padded']
pos_i = 0

count_ori = 0
count_256 = 0
count_AFP300 = 0
count_AFC300 = 0
count_AFP640 = 0
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

        image_path = os.path.join('EpicKitchen', fold, name)
        image = cv2.flip(cv2.imread(image_path), 1)
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
            elif fold == 'AFC640': count_AFP640 += 1
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
                elif fold == 'AFC640': count_AFP640 += 1
                else: count_pad += 1
                break
      
      if not have_results:
        no_detection += 1
        continue

      # else: 
      #   label1 = results.multi_handedness[0].classification[0].label
      #   label1 = label1.lower()
      #   if label1 in ori_path:
      #     count_ori += 1
          
      #   else:
      #     if len(results.multi_handedness) > 1:
      #       label2 = results.multi_handedness[1].classification[0].label
      #       label2 = label2.lower()
            
      #       if label2 in ori_path:
      #         count_ori += 1
            
      #       else: continue

      #     else: continue
        
      print(image.shape)
      image_height, image_width = 1080, 1920
      # if i == 63: 1/0
      annotated_image = image.copy()
      
      bb1 = [0, 0, 0, 0]
      # bb2 = [0, 0, 0, 0]

      d_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

      for ind, hand_landmarks in enumerate(results.multi_hand_landmarks):
        
        max_x = 0
        max_y = 0 
        min_x = 1920
        min_y = 1920

        label = results.multi_handedness[ind].classification[0].label
        label = label.lower()
        # if image.shape[0] == 300: 
        #   print(image_path, label)
        #   1/0
        # Necessary to not make a mess out of all that happens
        if label not in ori_path: continue

        # print(label)
        for mark in hand_landmarks.landmark:
          
          x_val = mark.x
          y_val = mark.y

          if 'AFC300' in image_path:
            if x_val * 1080 + 420 > max_x: max_x = x_val * 1080 + 420
            if x_val * 1080 + 420 < min_x: min_x = x_val * 1080 + 420

          else:
            if x_val * image_width > max_x: max_x = x_val * image_width
            if x_val * image_width < min_x: min_x = x_val * image_width

          # Due to the original resolution with padding 
          if image.shape[0] == 1920:
            if y_val * 1920 - 420 > max_y: max_y = y_val * 1920 - 420
            if y_val * 1920 - 420 < min_y: min_y = y_val * 1920 - 420

          elif image.shape[0] == 640:
            if y_val * 640 + 220 > max_y: max_y = y_val * 640 + 220
            if y_val * 640 + 220 < min_y: min_y = y_val * 640 + 220
          
          else:
            if y_val * image_height > max_y: max_y = y_val * image_height
            if y_val * image_height < min_y: min_y = y_val * image_height
  
        # wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        # thu_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
        # thu_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        # thu_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        # thu_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        # ind_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        # ind_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        # ind_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        # ind_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # mid_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        # mid_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        # mid_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        # mid_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        # ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        # ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
        # ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        
        # pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        # pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        # pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
        # pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
        # hand_components = [ wrist, 
        #                     thu_cmc, thu_ip, thu_ip, thu_tip, 
        #                     ind_mcp, ind_pip, ind_dip, ind_tip,
        #                     mid_mcp, mid_pip, mid_dip, mid_tip, 
        #                     ring_mcp, ring_pip, ring_dip, ring_tip, 
        #                     pinky_mcp, pinky_pip, pinky_dip, pinky_tip, 
        #                   ]
        
        # hc_x =  [c.x for c in hand_components]
        # hc_y =  [c.y for c in hand_components]
    
        # hc_x_norm = normalize(hc_x, 0, 1)
        # hc_y_norm = normalize(hc_y, 0, 1)
    
        x_dif = max_x - min_x
        y_dif = max_y - min_y
    
        # res_x = [image_width * x for x in hc_x]
        # res_y = [image_height * y for y in hc_y]
        
        
        # for hand_landmarks in results.multi_hand_landmarks:
        #   print('hand_landmarks:', hand_landmarks)
        #   print(
        #       f'Index finger tip coordinates: (',
        #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        #   )
        #   hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
    
          
    
        #   mp_drawing.draw_landmarks(
        #       annotated_image,
        #       hand_landmarks,
        #       mp_hands.HAND_CONNECTIONS,
        #       mp_drawing_styles.get_default_hand_landmarks_style(),
        #       mp_drawing_styles.get_default_hand_connections_style())
        
        # MP reverses the image, so we fix it in here.
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
        annotated_image = cv2.rectangle(annotated_image, ((min_x), int(np.round(min_y+420))), (max_x, int(np.round(max_y+420))), (255,255,0), 3)
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
      print('Detections in 640AfP resolution:', count_AFP640)
      print('Detections in padded original 1920x1080 resolution:', count_pad)
      print('No detections:', no_detection)
      # print(len(hc_x_norm), len(res_x))
      
      # for i in range(len(res_x)):
  
      #   annotated_image = cv2.putText(
      #     annotated_image, #numpy array on which text is written
      #     f'(*)', #text
      #     (int(np.round(res_x[i])), (int(np.round(res_y[i])))), #position at which writing has to start
      #     cv2.FONT_HERSHEY_SIMPLEX, #font family
      #     0.3, #font size
      #     (255, 255, 255, 255), #font color
      #     1) #font stroke
      if image.shape[0] == 1920:
        cv2.imwrite(
            'random/'+ name[:-4] + '.png', cv2.flip(annotated_image, 1))
        cv2.imwrite(
            'random/'+ name[:-4] + 'rev.png', annotated_image)
      # # Draw hand world landmarks.

      if not results.multi_hand_world_landmarks:
        continue
      # for hand_world_landmarks in results.multi_hand_world_landmarks:
      #   mp_drawing.plot_landmarks(
      #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

df.to_csv('EKdataMoreRes.csv', index=False)
