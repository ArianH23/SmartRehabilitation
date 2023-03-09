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

# For static images:
# image_name = 'precision3025.png'
# IMAGE_FILES = [f'frames/precision/{image_name}']
df = pd.DataFrame(columns=['col{}'.format(i) for i in range(1, 7)])
i = 0

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for path, subdirs, files in os.walk('frames'):
    for name in files:
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      i = i + 1
      print(i)
      print(name)

      image_path = os.path.join(path, name)
      image = cv2.flip(cv2.imread(image_path), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  
      # Print handedness and draw hand landmarks on the image.
      # print('Handedness:', results.multi_handedness)
      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
  
      max_x = 0
      max_y = 0 
      min_x = 1920
      min_y = 1920
      
      for hand_landmarks in results.multi_hand_landmarks:
        for mark in hand_landmarks.landmark:
              
          x_val = mark.x
          y_val = mark.y
  
          if x_val * image_width > max_x: max_x = x_val * image_width
          if x_val * image_width < min_x: min_x = x_val * image_width
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
      #                     ]
      
      # hc_x =  [c.x for c in hand_components]
      # hc_y =  [c.y for c in hand_components]
  
      # hc_x_norm = normalize(hc_x, 0, 1)
      # hc_y_norm = normalize(hc_y, 0, 1)
  
      # x_dif = max_x - min_x
      # y_dif = max_y - min_y
  
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

      width = max_x - min_x
      min_x = min_x-width
      
      height = int(np.round(max_y)) - int(np.round(min_y))
      

      d_list = [min_x, int(np.round(min_y)), width, height, name]

      if 'power' in name:
        d_list.append('power')
      
      elif 'precision' in name: d_list.append('precision')

      else: d_list.append('none')

      df.loc[len(df)] = d_list
      print(d_list)
      # annotated_image = cv2.rectangle(annotated_image, (int(np.round(min_x)), int(np.round(min_y))), (int(np.round(max_x)), int(np.round(max_y))), (255,255,0), 3)
      # print(len(hc_x_norm), len(res_x))
      # for i in range(len(res_x)):
  
      #   annotated_image = cv2.putText(
      #    annotated_image, #numpy array on which text is written
      #    f'({round(hc_x_norm[i],2)}, {round(hc_y_norm[i],2)})', #text
      #    (int(np.round(res_x[i])), (int(np.round(res_y[i])))), #position at which writing has to start
      #    cv2.FONT_HERSHEY_SIMPLEX, #font family
      #    0.3, #font size
      #    (255, 255, 255, 255), #font color
      #    1) #font stroke
  
      # cv2.imwrite(
      #     'mediapipe_photos/'+ image_name[:-4] + '.png', cv2.flip(annotated_image, 1))
      # cv2.imwrite(
      #     'mediapipe_photos/'+ image_name[:-4] + 'rev.png', annotated_image)
      # Draw hand world landmarks.

      if not results.multi_hand_world_landmarks:
        continue
      # for hand_world_landmarks in results.multi_hand_world_landmarks:
      #   mp_drawing.plot_landmarks(
      #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

df.to_csv('dataset_dist_to_min/mp_bb.csv', index=False)
