# 输出由模型得到的特征向量，并分别计算类间，和类内的向量之间的欧式距离，得出误差率。
model_ft = models.resnet152(pretrained=True)

model = model_ft


# pretrained_dict=model.state_dict()
# print(pretrained_dict['features.0.weight'])
# model.load_state_dict(torch.load('resnet18_Fingervein.pkl'))
# print(model.state_dict()['features.0.weight'])

def get_vectorarray(model, dset_loaders):
    model.eval()
    k = 0
    m = 0
    data_features15_Inter = np.ndarray(shape=(86, 1000))  # 类间
    data_features15_Inter[:, :] = 0

    data_features1024_Intra = np.ndarray(shape=(86, 15, 1000))  # 类内
    data_features1024_Intra[:, :, :] = 0
    for j, data in enumerate(dset_loaders):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        # print(outputs,labels)
        if k != 15:
            data_features15_Inter[m, :] = data_features15_Inter[m, :] + outputs.data.numpy()
            data_features1024_Intra[m, k, :] = outputs.data.numpy()
            k = k + 1
        else:
            m = m + 1
            data_features15_Inter[m, :] = data_features15_Inter[m, :] + outputs.data.numpy()
            data_features1024_Intra[m, 0, :] = outputs.data.numpy()
            k = 1
    data_features15_Inter[:, :] = data_features15_Inter[:, :] / 15
    return data_features15_Inter, data_features1024_Intra


data_Inter, data_Intra = get_vectorarray(model, newdata_loader)
print('end')


def get_result(data_Inter, data_Intra):
    count_Inter = 0
    num_Inter = 0
    disum_Inter = 0
    dist_Interlist = []
    minidis_Inter = np.linalg.norm(data_Inter[0] - data_Inter[1])
    maxdis_Inter = np.linalg.norm(data_Inter[0] - data_Inter[1])

    count_Intra = 0
    num_Intra = 0
    disum_Intra = 0
    dist_Intralist = []
    maxdis_Intra = np.linalg.norm(data_Intra[0, 0, :] - data_Intra[0, 1, :])
    minidis_Intra = np.linalg.norm(data_Intra[0, 0, :] - data_Intra[0, 1, :])
    for i in range(86):
        for n in range(i + 1, 86):  # 32*32
            dist = np.linalg.norm(data_Inter[i] - data_Inter[n])
            dist_Interlist.append(dist)
            if minidis_Inter > dist:
                minidis_Inter = dist
            if maxdis_Inter < dist:
                maxdis_Inter = dist
            disum_Inter += dist
            count_Inter += 1
            if dist <= 9:
                num_Inter = num_Inter + 1

    for i in range(86):
        for n in range(15):
            for s in range(n + 1, 15):  # 64*4*4
                dist = np.linalg.norm(data_Intra[i, n, :] - data_Intra[i, s, :])
                dist_Intralist.append(dist)
                if maxdis_Intra < dist:
                    maxdis_Intra = dist
                if minidis_Intra > dist:
                    minidis_Intra = dist
                disum_Intra = disum_Intra + dist
                count_Intra = count_Intra + 1
                if dist > 9:
                    num_Intra = num_Intra + 1

    print('类间总数为 {}\n错误数为{}\n错误率为{}\n最大类间距为{}\n最小类间距为{}\n平均距离为{}'
          .format(count_Inter, num_Inter, num_Inter / count_Inter, maxdis_Inter, minidis_Inter,
                  disum_Inter / count_Inter))
    print('——' * 10)
    print('类内总数为 {}\n错误数为{}\n错误率为{}\n最大类内距为{}\n最小类内距为{}\n平均距离为{}'
          .format(count_Intra, num_Intra, num_Intra / count_Intra, maxdis_Intra, minidis_Intra,
                  disum_Intra / count_Intra))
    return dist_Interlist, dist_Intralist


dist_Inter_list, dist_Intra_list = get_result(data_Inter, data_Intra)



import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('dist')
plt.ylabel("frequence")
plt.title("Inter class distance")
#n, bins, patches = plt.hist(dist_Inter_list,len(dist_Inter_list), normed=0, facecolor='green', alpha=0.75)
plt.hist(dist_Inter_list,len(dist_Inter_list), normed=0, facecolor='green', alpha=0.75)
plt.show()
plt.xlabel('dist')
plt.ylabel("frequence")
plt.title("Intra class distance")
plt.hist(dist_Intra_list,len(dist_Intra_list), normed=0, facecolor='green', alpha=0.75)
plt.show()