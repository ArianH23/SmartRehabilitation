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

possible_folders = ['resultFrames']
pos_i = 0

count_ori = 0
count_256 = 0
count_AFP300 = 0
count_AFC300 = 0
count_pad = 0
no_detection = 0

dfread = pd.read_csv('../data/yale_hand_labels2_adapted.csv')


with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.15) as hands:
  for index, row in dfread.iterrows():
    
    files = row['Filename']

    i = i + 1
      

    have_results = False
    for fold in possible_folders:

        image_path = files  #os.path.join('EpicKitchen', fold, name)
        # print(files, 'va')
        image = cv2.flip(cv2.imread(image_path), 1)
        # image_flipped = cv2.imread(image_path)
        
        annotated_image = cv2.flip(cv2.imread(image_path), 1)
        # annotated_image_flipped = cv2.imread(image_path)

        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image_height, image_width = 480, 640

    res = results
    
    bb1 = [0, 0, 0, 0]
    # bb2 = [0, 0, 0, 0]

    d_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if res.multi_hand_landmarks== None: continue

    # You only want to detct right hands, as those are the only hands that have labels
    rDect = False
    for ind, hand_landmarks in enumerate(res.multi_hand_landmarks):
        if rDect: continue
        max_x = 0
        max_y = 0 
        min_x = 640
        min_y = 640

        label = res.multi_handedness[ind].classification[0].label
        label = label.lower()
        if label =='left': continue
        else: rDect = True

        # print(label)
        for mark in hand_landmarks.landmark:
        
            x_val = mark.x
            y_val = mark.y

            if 'AFC' in image_path:
                pass
                if x_val * 1080 + 420 > max_x: max_x = x_val * 1080 + 420
                if x_val * 1080 + 420 < min_x: min_x = x_val * 1080 + 420

            else:
                if x_val * image_width > max_x: max_x = x_val * image_width
                if x_val * image_width < min_x: min_x = x_val * image_width

            # Due to the original resolution with padding 
            if 'AFP' in image_path:
                pass
                if y_val * 1920 - 420 > max_y: max_y = y_val * 1920 - 420
                if y_val * 1920 - 420 < min_y: min_y = y_val * 1920 - 420
            
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

        max_x = min(640, max_x)
        max_y = min(480, max_y)

        width = max_x - min_x
        
        height = int(np.round(max_y)) - int(np.round(min_y))

        bb1 = [min_x, int(np.round(min_y)), width, height, label]


        # annotated_image = cv2.flip(annotated_image,1)
        # annotated_image = cv2.rectangle(annotated_image, ((min_x), int(np.round(min_y))), (max_x, int(np.round(max_y))), (255,255,0), 3)
        # annotated_image = cv2.flip(annotated_image,1)

        # if i == 0:
        annotated_image = cv2.flip(annotated_image,1)
        annotated_image = cv2.rectangle(annotated_image, ((min_x), int(np.round(min_y))), (max_x, int(np.round(max_y))), (255,255,0), 3)
        annotated_image = cv2.flip(annotated_image,1)
        # cv2.imwrite(
        # '../photostest/'+ files[15:-4] + '.png', cv2.flip(annotated_image, 1))
        # else:
        #     annotated_image_flipped = cv2.flip(annotated_image_flipped,1)
        #     annotated_image_flipped = cv2.rectangle(annotated_image_flipped, ((min_x), int(np.round(min_y))), (max_x, int(np.round(max_y))), (255,255,0), 3)
        #     annotated_image_flipped = cv2.flip(annotated_image_flipped,1)
        #     cv2.imwrite(
        #     '../photostest/'+ files[15:-4] + 'i.png', cv2.flip(annotated_image_flipped, 1))

    # if bb1==[0,0,0,0] : continue
    # if i == 0:
    d_list = bb1 + [files]
    # else:
    #     d_list = bb1 + ['i'+files]

    # print(d_list)
    if 'power' in row['SmallCategories']:
        d_list.append('power')
    
    elif 'precision' in row['SmallCategories']: d_list.append('precision')

    else: d_list.append('none')
    if len(d_list) == len(df.columns):
        df.loc[len(df)] = d_list

    print(df)
    print('Detections in 256 resolution:', count_256)
    print('Detections in 300AfP resolution:', count_AFP300)
    print('Detections in 300AfC resolution:', count_AFC300)
    print('Detections in original 1920x1080 resolution:', count_ori)
    print('Detections in 1920Afp resolution:', count_pad)
    print('No detections:', no_detection)
    print(fold)
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
      # if image.shape[0] == 1920:
    #   cv2.imwrite(
    #       'randomBB/'+ name[:-4] + '.png', cv2.flip(annotated_image, 1))
      # cv2.imwrite(
      #     'random/'+ name[:-4] + 'rev.png', annotated_image)
      # # Draw hand world landmarks.

    if not results.multi_hand_world_landmarks:
        continue
      # for hand_world_landmarks in results.multi_hand_world_landmarks:
      #   mp_drawing.plot_landmarks(
      #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

df.to_csv('yale_mp.csv', index=False)
