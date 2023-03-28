from pathlib import Path
import numpy as np
import struct
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def get_median(lst):
    sorted_lst = sorted(lst)
    length = len(sorted_lst)
    middle_index = length // 2
    
    if length % 2 != 0:
        # length is odd, so return middle element
        return sorted_lst[middle_index]
    else:
        # length is even, so return average of middle two elements
        return (sorted_lst[middle_index] + sorted_lst[middle_index - 1]) / 2


def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)
        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale

image = read_pfm('frames_simu_partial_gs/power/simulation00113.png-dpt_beit_large_512.pfm')
image = [row[1:1800] for row in image[1:500]] # 1920 x 1080

# plt.imshow(image) 
# plt.show()

df = pd.read_csv('data_for_gs/MPandYOLOandLM_dist2Hands_corrections.csv')
df2 = pd.read_csv('data_for_gs/simu_yolo_lm.csv')

df = pd.concat([df,df2], ignore_index=True)
df_fin = pd.DataFrame(columns=list(df.columns) + ['lh_depth_dist', 'rh_depth_dist'])

for index, row in df.iterrows():
    aux_row = list(row) + [0,0]

    folder = 'frames_simu_partial_gs' if 'simulation' in row['picture_name'] else 'frames_salad_gs'
    grasp = row['grasp']

    pfm = row['picture_name'] + '-dpt_beit_large_512.pfm'

    path = folder + '/' + grasp + '/' + pfm
    image = read_pfm(path)

    lh_x1 = row['l_hand_min_x']
    lh_y1 = row['l_hand_min_y']
    lh_x2 = lh_x1 + row['l_hand_width']
    lh_y2 = lh_y1 + row['l_hand_height']

    rh_x1 = row['r_hand_min_x']
    rh_y1 = row['r_hand_min_y']
    rh_x2 = rh_x1 + row['r_hand_width']
    rh_y2 = rh_y1 + row['r_hand_height']

    o_x1 = row['object_min_x']
    o_y1 = row['object_min_y']
    o_x2 = o_x1 + row['object_width']
    o_y2 = o_y1 + row['object_height']

    if lh_x1 == 1920: lh_x1= lh_x1-1
    if lh_y1 == 1080: lh_y1= lh_y1-1
    if lh_x2 == 1920: lh_x2= lh_x2-1
    if lh_y2 == 1080: lh_y2= lh_y2-1

    if rh_x1 == 1920: rh_x1= rh_x1-1
    if rh_y1 == 1080: rh_y1= rh_y1-1
    if rh_x2 == 1920: rh_x2= rh_x2-1
    if rh_y2 == 1080: rh_y2= rh_y2-1

    if o_x1 == 1920: o_x1= o_x1-1
    if o_y1 == 1080: o_y1= o_y1-1
    if o_x2 == 1920: o_x2= o_x2-1
    if o_y2 == 1080: o_y2= o_y2-1
    
    bbLH_m = [row[lh_x1: lh_x2] for row in image[lh_y1:lh_y2]]
    bbRH_m = [row[rh_x1: rh_x2] for row in image[rh_y1:rh_y2]]
    bbO_m =  [row[o_x1: o_x2]   for row in image[o_y1:o_y2]]

    bbLH = []
    bbRH = []
    bbO = []

    for sublist in bbLH_m:
        for item in sublist:
            bbLH.append(item)


    for sublist in bbRH_m:
        for item in sublist:
            bbRH.append(item)

    for sublist in bbO_m:
        for item in sublist:
            bbO.append(item)

    if row['l_hand_width'] == 0:
        bbLH = [image[lh_y1][lh_x1]
]
    if row['r_hand_width'] == 0:
        bbRH = [image[rh_y1][rh_x1]]

    if row['object_width'] == 0:
        bbO = [image[o_y1][o_x1]]

    med_LH = get_median(bbLH)
    med_RH = get_median(bbRH)
    med_O = get_median(bbO)

    aux_row[-2] = med_O - med_LH
    aux_row[-1] = med_O - med_RH

    df_fin.loc[len(df_fin)] = aux_row
    print(aux_row[-2])
    print(aux_row[-1])
    print()