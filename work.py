import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.goods_utils import make_embeddings, make_knn
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from utils.goods_utils import *
from network.goods_net import EmbedNetworkResnet
from torch.utils.data import DataLoader
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def setNum(num):
    lst = []
    for i in range(num):
        lst.append(i+1)
    return lst

def setNumCenter(num):
    return setNum(num)


def setNumNeighbors(num):
    return setNum(num)




def testCenterAndNeighbors(register_loader,model ,val_loader,setMaxCenter,setMaxNeighbors):

    NumCenter = setNumCenter(setMaxCenter)
    NumNeighbors = setNumNeighbors(setMaxNeighbors)
    ScoresArr = np.zeros(shape=(setMaxCenter,setMaxNeighbors))
    ScoresArr = np.array(ScoresArr)
    for num_centers in NumCenter:

        for num_neighbors in NumNeighbors:
            score = test_embed_multi(model, register_loader, val_loader, num_centers, num_neighbors)
            ScoresArr[num_centers-1][num_neighbors-1] = score
    m = np.argmax(ScoresArr)
    NumCenter,NumNeighbors = divmod(m, ScoresArr.shape[1])

    #######
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # X = np.arange(1, ScoresArr.shape[0], 0.1)
    # Y = np.arange(1, ScoresArr.shape[1], 0.1)
    # X, Y = np.meshgrid(X, Y)
    # ax.plot_surface(X, Y, ScoresArr, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    return NumCenter+1,NumNeighbors+1,ScoresArr


#输入模型，注册数据集
def test_embed_multi(model, register_loader, val_loader, num_centers, num_neighbors):


    valid_embedding, valid_labels, paths, ids = make_embeddings_path(val_loader, model)
    valid_embedding = normalize(valid_embedding)
    # valid_embedding, valid_labels = make_embeddings(val_loader, model)
    # valid_embedding = normalize(valid_embedding)
    id_to_label = register_loader.dataset.class_to_idx
    label_to_id = {a: b for b, a in id_to_label.items()}

    knn = make_knn(register_loader, model, num_centers=num_centers, num_neighbors=num_neighbors)
    accuracy = knn.score(valid_embedding, valid_labels)
    predicts = knn.predict(valid_embedding)
    acc_dict = {}
    for label in set(valid_labels):
        all = np.count_nonzero(valid_labels == label)
        correct = np.count_nonzero(predicts[valid_labels == label] == label)
        id = label_to_id[int(label)]
        if all != 0:
            acc = correct / all
        else:
            acc = 1
        acc_dict[id] = acc
    return accuracy



if __name__ == '__main__':
    model = EmbedNetworkResnet(1475)
    model_path = 'project/model'
    dataset_name = '20190224'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.Parameter = checkpoint['cls_dict']
    if cuda:
        model = torch.nn.parallel.DataParallel(model)
        model.cuda()

    dataset_path = r'/home/sl/.shelf/classification/knn_models/register_classified_cropped_images'
    register_dataset = ImageFolderWithPaths(dataset_path, transform=dev_test_transform,
                                            loader=exampler_loader)

    register_loader = DataLoader(register_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    val_loader = DataLoader(register_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)




    setMaxCenter=2
    setMaxNeighbors=3
    NumCenter, NumNeighbors, ScoresArr = testCenterAndNeighbors(register_loader, model, val_loader, setMaxCenter,
                                                                setMaxNeighbors)
    print("NumCenter:",NumCenter," NumNeighbors:", NumNeighbors)
    print(ScoresArr)
