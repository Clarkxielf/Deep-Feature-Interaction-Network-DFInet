import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch

color = ['red', 'orange', 'blue', 'green', 'fuchsia', 'black', 'yellow']



'''Mechanical Registration'''
inspection_data = sio.loadmat('blade1_S1_Inspection_data.mat')['Inspection_data'][:, :2]

TheTa = (150/180) * np.pi
Rot_sr = np.array([[np.cos(TheTa), -np.sin(TheTa)],
                   [np.sin(TheTa), np.cos(TheTa)]]).squeeze()
inspection_data = inspection_data @ Rot_sr.transpose(-1, -2)

plt.scatter(inspection_data[:, 0],
            inspection_data[:, 1],
            s=5, facecolors='none', edgecolors=color[0])
plt.show()



'''Index of raw view'''
distance = 0.5
segmented_data = {}
data_list = []
segmented_num = 0
for idx in range(0, inspection_data.shape[0]-1):

    if idx!=inspection_data.shape[0]-2:
        if np.sqrt(((inspection_data[idx+1]-inspection_data[idx])**2).sum()) < distance:
            data_list.append(inspection_data[idx][None, ...])
        else:
            data_list.append(inspection_data[idx][None, ...])
            print(idx+1)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(inspection_data[idx][None, ...])
        data_list.append(inspection_data[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

plt.show()



'''Interpolation'''
segmented_data0 = inspection_data[:1771]
distance = 0.01
for _ in range(3):
    new_data = []
    for idx in range(1, segmented_data0.shape[0]):
        if np.sqrt(((segmented_data0[idx]-segmented_data0[idx-1])**2).sum()) < distance:
            new_data.append(segmented_data0[idx-1])
        else:
            new_data.append(segmented_data0[idx - 1])
            interpolation = (segmented_data0[idx]+segmented_data0[idx-1])/2
            new_data.append(interpolation)
    new_data.append(segmented_data0[-1])
    segmented_data0 = np.stack(new_data, axis=0)

distance = 0.01
segmented_data = {}
data_list = []
segmented_num = 0
for idx in range(0, segmented_data0.shape[0]-1):

    if idx!=segmented_data0.shape[0]-2:
        if np.sqrt(((segmented_data0[idx+1]-segmented_data0[idx])**2).sum()) < distance:
            data_list.append(segmented_data0[idx][None, ...])
        else:
            data_list.append(segmented_data0[idx][None, ...])
            print(idx+1)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(segmented_data0[idx][None, ...])
        # data_list.append(segmented_data0[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

plt.show()


segmented_data1 = inspection_data[1771:3951]
distance = 0.01
for _ in range(5):
    new_data = []
    for idx in range(1, segmented_data1.shape[0]):
        if np.sqrt(((segmented_data1[idx]-segmented_data1[idx-1])**2).sum()) < distance:
            new_data.append(segmented_data1[idx-1])
        else:
            new_data.append(segmented_data1[idx - 1])
            interpolation = (segmented_data1[idx]+segmented_data1[idx-1])/2
            new_data.append(interpolation)
    new_data.append(segmented_data1[-1])
    segmented_data1 = np.stack(new_data, axis=0)

distance = 0.01
segmented_data = {}
data_list = []
segmented_num = 0
for idx in range(0, segmented_data1.shape[0]-1):

    if idx!=segmented_data1.shape[0]-2:
        if np.sqrt(((segmented_data1[idx+1]-segmented_data1[idx])**2).sum()) < distance:
            data_list.append(segmented_data1[idx][None, ...])
        else:
            data_list.append(segmented_data1[idx][None, ...])
            print(idx+1)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(segmented_data1[idx][None, ...])
        data_list.append(segmented_data1[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

plt.show()


segmented_data2 = inspection_data[3951:]
distance = 0.01
for _ in range(2):
    new_data = []
    for idx in range(1, segmented_data2.shape[0]):
        if np.sqrt(((segmented_data2[idx]-segmented_data2[idx-1])**2).sum()) < distance:
            new_data.append(segmented_data2[idx-1])
        else:
            new_data.append(segmented_data2[idx - 1])
            interpolation = (segmented_data2[idx]+segmented_data2[idx-1])/2
            new_data.append(interpolation)
    new_data.append(segmented_data2[-1])
    segmented_data2 = np.stack(new_data, axis=0)

distance = 0.01
segmented_data = {}
data_list = []
segmented_num = 0
for idx in range(0, segmented_data2.shape[0]-1):

    if idx!=segmented_data2.shape[0]-2:
        if np.sqrt(((segmented_data2[idx+1]-segmented_data2[idx])**2).sum()) < distance:
            data_list.append(segmented_data2[idx][None, ...])
        else:
            data_list.append(segmented_data2[idx][None, ...])
            print(idx+1)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(segmented_data2[idx][None, ...])
        data_list.append(segmented_data2[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

plt.show()


'''Equidistant sampling'''
distance = 0.02
new_data = []
for segmented_data in [segmented_data0, segmented_data1, segmented_data2]:
    resample_data = []
    idx = 0
    for num in range(segmented_data.shape[0]):
        i = 0
        while np.sqrt(((segmented_data[idx+i]-segmented_data[idx])**2).sum()) < distance:
            i = i+1
            if idx+i == segmented_data.shape[0]-1:
                break

        resample_data.append(segmented_data[idx])
        if idx+i == segmented_data.shape[0] - 1:
            break

        idx = idx + i

    segmented_data = np.stack(resample_data, axis=0)
    new_data.append(segmented_data)
Inference_data = np.concatenate(new_data, axis=0)

'''Index of sample view'''
distance = 0.03
segmented_data = {}
data_list = []
segmented_num = 0
for idx in range(0, Inference_data.shape[0]-1):

    if idx!=Inference_data.shape[0]-2:
        if np.sqrt(((Inference_data[idx+1]-Inference_data[idx])**2).sum()) < distance:
            data_list.append(Inference_data[idx][None, ...])
        else:
            data_list.append(Inference_data[idx][None, ...])
            print(idx+1)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(Inference_data[idx][None, ...])
        data_list.append(Inference_data[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

plt.show()


'''Dataset'''
segmented_data0 = new_data[0]
segmented_data1 = new_data[1]
segmented_data2 = new_data[2]

all_Src = []
all_Ref = []
all_Rot_mat = []
all_Translation = []
all_TheTa = []

src_num = 256
ref_num = 272
TheTa0 = np.array([1/180]) * np.pi
Translation0 = np.array([0.3, 0.1])
Rot_sr0 = np.array([[np.cos(TheTa0), -np.sin(TheTa0)],
                   [np.sin(TheTa0), np.cos(TheTa0)]]).squeeze()

src = segmented_data0[-src_num:]
index_head = ((src[0, :]-segmented_data1)**2).sum(-1).argmin()
index_tail = ((src[-1, :]-segmented_data1)**2).sum(-1).argmin()
index_deviation = (ref_num-(index_tail-index_head))//2
if index_head-index_deviation>=0:
    ref = segmented_data1[index_head-index_deviation:index_head-index_deviation+ref_num]
else:
    ref = segmented_data1[:ref_num, :]
# ref = segmented_data1[:ref_num]
# index_head = ((ref[0, :]-segmented_data0)**2).sum(-1).argmin()
# index_tail = ((ref[-1, :]-segmented_data0)**2).sum(-1).argmin()
# index_deviation = ((index_tail-index_head)-src_num)//2
# if index_head+index_deviation+src_num<segmented_data.shape[0]:
#     src = segmented_data1[index_head+index_deviation:index_head+index_deviation+src_num]
# else:
#     src = segmented_data1[-src_num:, :]


all_Src.append(src[None, ...])
all_Ref.append(ref[None, ...])
all_TheTa.append(TheTa0[None, ...])
all_Rot_mat.append(Rot_sr0[None, ...])
all_Translation.append(Translation0[None, ...])

plt.scatter(src[:, 0],
            src[:, 1],
            s=50, facecolors='none', edgecolors=color[0])
plt.scatter(ref[:, 0],
            ref[:, 1],
            s=20, facecolors='none', edgecolors=color[1])
plt.scatter(Inference_data[:, 0],
            Inference_data[:, 1],
            s=2, facecolors='none', edgecolors=color[2])
plt.show()


src_num = 256*3
ref_num = 272*3
TheTa1 = np.array([-2/180]) * np.pi
Translation1 = np.array([-0.3, 0.2])
Rot_sr1 = np.array([[np.cos(TheTa1), -np.sin(TheTa1)],
                   [np.sin(TheTa1), np.cos(TheTa1)]]).squeeze()

src = segmented_data1[-src_num:]
index_head = ((src[0, :]-segmented_data2)**2).sum(-1).argmin()
index_tail = ((src[-1, :]-segmented_data2)**2).sum(-1).argmin()
index_deviation = (ref_num-(index_tail-index_head))//2
if index_head-index_deviation>=0:
    ref = segmented_data2[index_head-index_deviation:index_head-index_deviation+ref_num]
else:
    ref = segmented_data2[:ref_num, :]
# ref = segmented_data2[:ref_num]
# index_head = ((ref[0, :]-segmented_data1)**2).sum(-1).argmin()
# index_tail = ((ref[-1, :]-segmented_data1)**2).sum(-1).argmin()
# index_deviation = ((index_tail-index_head)-src_num)//2
# if index_head+index_deviation+src_num<segmented_data1.shape[0]:
#     src = segmented_data1[index_head+index_deviation:index_head+index_deviation+src_num]
# else:
#     src = segmented_data1[-src_num:, :]

new_data = []
i = 0
num = 3
while num * i < src.shape[0]:
    new_data.append(src[num * i])
    i += 1
src = np.stack(new_data, axis=0)

new_data = []
i = 0
num = 3
while num * i < ref.shape[0]:
    new_data.append(ref[num * i])
    i += 1
ref = np.stack(new_data, axis=0)


all_Src.append(src[None, ...])
all_Ref.append(ref[None, ...])
all_TheTa.append(TheTa1[None, ...])
all_Rot_mat.append(Rot_sr1[None, ...])
all_Translation.append(Translation1[None, ...])

plt.scatter(src[:, 0],
            src[:, 1],
            s=50, facecolors='none', edgecolors=color[0])

plt.scatter(ref[:, 0],
            ref[:, 1],
            s=20, facecolors='none', edgecolors=color[1])
plt.scatter(Inference_data[:, 0],
            Inference_data[:, 1],
            s=2, facecolors='none', edgecolors=color[2])
plt.show()


all_Src = np.concatenate(all_Src, 0)   # B*N*2
all_Ref = np.concatenate(all_Ref, 0)   # B*N*2
all_Rot_mat = np.concatenate(all_Rot_mat, 0)   # B*2*2
all_Translation = np.concatenate(all_Translation, 0)   # B*2
all_TheTa = np.concatenate(all_TheTa, 0)   # B*1


# sio.savemat('result/blade1_S1_inference_data_test0.mat',
#             {'Src': all_Src,
#              'Ref': all_Ref,
#              'Rot_mat': all_Rot_mat,
#              'Translation': all_Translation,
#              'TheTa': all_TheTa})
#
# sio.savemat('result/blade1_S1_preprocess.mat',
#             {'inspection_data': inspection_data})


distance = 0.5
all_Segmented_idx = []

segmented_data = {}
data_list = []
segmented_num = 0
all_Segmented_idx.append(0)
for idx in range(0, inspection_data.shape[0]-1):

    if idx!=inspection_data.shape[0]-2:
        if np.sqrt(((inspection_data[idx+1]-inspection_data[idx])**2).sum()) < distance:
            data_list.append(inspection_data[idx][None, ...])
        else:
            data_list.append(inspection_data[idx][None, ...])
            print(idx+1)
            all_Segmented_idx.append(idx+1)
            segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

            plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                        segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                        s=5, facecolors='none', edgecolors=color[segmented_num])

            segmented_num = segmented_num + 1
            data_list = []

    else:
        data_list.append(inspection_data[idx][None, ...])
        data_list.append(inspection_data[idx+1][None, ...])
        segmented_data['segmented_{}'.format(segmented_num)] = np.concatenate(data_list, axis=0)

        plt.scatter(segmented_data['segmented_{}'.format(segmented_num)][:, 0],
                    segmented_data['segmented_{}'.format(segmented_num)][:, 1],
                    s=5, facecolors='none', edgecolors=color[segmented_num])

        all_Segmented_idx.append(idx+1+1)

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0, wspace=0)
plt.show()

# all_Segmented_idx.append(0)
# all_Segmented_idx = np.stack(all_Segmented_idx, 0)
# sio.savemat('result/blade1_S1_inference_data_Segmented_idx.mat',
#             {'all_Segmented_idx': all_Segmented_idx})
