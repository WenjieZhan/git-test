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
import time
import os
import pandas as pd
import json
use_embeddings_to_generate_knn=False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"




def testCenterAndNeighbors(register_loader,model ,val_loader,setMaxCenter,setMaxNeighbors,use_embeddings_to_generate_knn):

    Centers = [i for i in range(1,setMaxCenter)]
    NumCenter = len(Centers)
    Neighbors = [i for i in range(1,setMaxNeighbors,2)]
    NumNeighbor = len(Neighbors)
    ScoresArr = np.zeros(shape=(NumCenter,NumNeighbor))

    ScoresArr = np.array(ScoresArr)

    if use_embeddings_to_generate_knn:
        produce_scores = use_embedding_to_make_knn
    else:
        produce_scores = test_embed_multi
    print(ScoresArr.shape)
    i=0
    for center in Centers:
        j=0
        for neighbor in Neighbors:
            score = produce_scores(model, register_loader, val_loader, center, neighbor)
            ScoresArr[i][j] = score
            j+=1
        i+=1

    data_index = ["center:"+str(i) for i in Centers]
    data_columns = ["neighbor:" + str(i) for i in Neighbors]
    index = pd.Index(data=data_index)
    columns = pd.Index(data=data_columns)
    m = np.argmax(ScoresArr)
    NumCenter,NumNeighbors = divmod(m, ScoresArr.shape[1])

    ScoresArr = pd.DataFrame(ScoresArr, index=index, columns=columns)
    ScoresArr.to_excel("/home/sl/.shelf/classification/knn_models/"+time.strftime("%Y%m%d",time.localtime())+"/NumCenterNumNeighbors.xlsx")
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

def use_embedding_to_make_knn(knn_model,path= None, num_centers=5, num_neighbors=5):
    # embeddings, labels, paths, ids = make_embeddings_path(test_register_loader, model)
    # embeddings = normalize(embeddings)

    with open(os.path.expanduser('~') + r"/.shelf/classification/knn_models/" + time.strftime("%Y%m%d",
                                                                                              time.localtime()) + "/register_embeddings.pkl",
              'rb') as f:
        embeddings=pickle.load( file=f)
    with open(os.path.expanduser('~') + "/.shelf/classification/knn_models/" + time.strftime("%Y%m%d",
                                                                                             time.localtime()) + "/index_to_label.json",
              'r') as f:
        labels=json.load(fp=f)

    print('num_center:{}'.format(num_centers))
    label_set = set(labels)
    num_class = len(label_set)
    print('num_class:{}'.format(num_class))
    all_centers = np.zeros((num_class, num_centers, args.embed_dim))
    all_labels = np.zeros((num_class, num_centers))

    print('start finding centers')
    for i in range(len(label_set)):
        label = list(label_set)[i]
        embedding_test = embeddings[labels == label]
        kmeans = KMeans(n_clusters=num_centers)
        kmeans = kmeans.fit(embedding_test)
        centers = kmeans.cluster_centers_

        all_centers[i] = centers
        all_labels[i] = label

    all_labels_reshape = all_labels.reshape(-1)
    all_centers_reshape = all_centers.reshape(-1, args.embed_dim)


    print('num_neighbors{}'.format(num_neighbors))
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, n_jobs=-1,)# weights='distance'
    print('Start generating KNN model')
    knn = knn.fit(all_centers_reshape, all_labels_reshape)
    if knn_model_path:
        joblib.dump(knn, knn_model_path)
    return knn



def save_knn(model, register_loader,knn_model_path, num_centers, num_neighbors):
    knn = make_knn(register_loader, model,knn_model_path, num_centers=num_centers, num_neighbors=num_neighbors)


if __name__ == '__main__':



    model = EmbedNetworkResnet(1229)
    model_path = os.path.expanduser('~') + '/.shelf/classification/models/20190313/epoch_55'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.Parameter = checkpoint['cls_dict']
    if cuda:
        model = torch.nn.parallel.DataParallel(model)
        model.cuda()

    if not use_embeddings_to_generate_knn:

        dataset_path = r'/home/sl/.shelf/classification/knn_models/register_classified_cropped_images'
        register_dataset = ImageFolderWithPaths(dataset_path, transform=dev_test_transform,
                                                loader=exampler_loader)

        register_loader = DataLoader(register_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        val_loader = DataLoader(register_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    setMaxCenter=2
    setMaxNeighbors=5
    NumCenter, NumNeighbors, ScoresArr = testCenterAndNeighbors(register_loader, model, val_loader, setMaxCenter,
                                                                setMaxNeighbors,use_embeddings_to_generate_knn)
    print("NumCenter:",NumCenter,"NumNeighbors:", NumNeighbors)
    print(ScoresArr)
    save_path = os.path.expanduser('~') +"/.shelf/classification/knn_models/"+time.strftime("%Y%m%d",time.localtime())+"/"
    knn_model_path =save_path +"knn_model_Center-"+str(NumCenter)+"_Neighbor-"+str(NumNeighbors)
    save_knn(model, register_loader,knn_model_path , NumCenter, NumNeighbors)
