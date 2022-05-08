'''生成理论数据集'''
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
color = ['red', 'orange', 'blue', 'green', 'fuchsia', 'black', 'yellow']

raw_data = pd.read_excel('./GMD6429584P1003_-53.08-0.02-sort.xlsx', sheet_name='Summary')
raw_data = raw_data.values[:, 1:]

TheTa = (90/180) * np.pi
Rot_sr = np.array([[np.cos(TheTa), -np.sin(TheTa)],
                   [np.sin(TheTa), np.cos(TheTa)]]).squeeze()
raw_data = raw_data @ Rot_sr.transpose(-1, -2)

plt.scatter(raw_data[1500:2700, 0], raw_data[1500:2700, 1], s=20, facecolors='none', edgecolors=color[0])
plt.scatter(raw_data[:, 0], raw_data[:, 1], s=2, facecolors='none', edgecolors=color[1])
plt.show()


num_sample = 16000
src_num_sample_points = 256
ref_num_sample_points = 272
extend_num = np.random.randint(0, ref_num_sample_points-src_num_sample_points, 1)

all_Src = []
all_Ref = []
all_Rot_mat = []
all_Translation = []
for num in range(num_sample):

    base_num = np.random.randint(1500, 2700, 1)
    src = raw_data[int(base_num-src_num_sample_points):int(base_num), :2]
    raw_ref = raw_data[int(base_num-ref_num_sample_points+extend_num):int(base_num+extend_num), :2]

    assert src.shape[0] == src_num_sample_points and raw_ref.shape[0] == ref_num_sample_points


    TheTa = np.random.random(1) * np.pi/72
    Translation = 2 * (np.random.random((1, 2)) - 0.5) * 0.2
    Rot_sr = np.array([[np.cos(TheTa), -np.sin(TheTa)],
                       [np.sin(TheTa), np.cos(TheTa)]]).squeeze()

    ref = raw_ref @ Rot_sr.transpose(-1, -2) + Translation

    # plt.scatter(src[:, 0], src[:, 1],  s=50, facecolors='none', edgecolors=color[0])
    # plt.scatter(raw_ref[:, 0], raw_ref[:, 1],  s=20, facecolors='none', edgecolors=color[1])
    # plt.scatter(ref[:, 0], ref[:, 1],  s=20, facecolors='none', edgecolors=color[2])
    # plt.scatter(raw_data[:, 0], raw_data[:, 1], s=2, facecolors='none', edgecolors=color[3])
    # plt.show()

    all_Src.append(src[None, ...])
    all_Ref.append(ref[None, ...])
    all_Rot_mat.append(Rot_sr[None, ...])
    all_Translation.append(Translation)


all_Src = np.concatenate(all_Src, 0)   # B*N*2
all_Ref = np.concatenate(all_Ref, 0)   # B*N*2
all_Rot_mat = np.concatenate(all_Rot_mat, 0)   # B*2*2
all_Translation = np.concatenate(all_Translation, 0)   # B*2

sio.savemat('result/0.02_blade1_64data_2d_train0.mat',
            {'Src': all_Src,
             'Ref': all_Ref,
             'Rot_mat': all_Rot_mat,
             'Translation': all_Translation})