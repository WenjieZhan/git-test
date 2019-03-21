"""
Code for generating knn model
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from utils.goods_utils import make_embeddings, make_knn
from sklearn.preprocessing import normalize
from network.goods_net import EmbedNetworkResnet
from utils.goods_utils import exampler_loader, dev_test_transform, goods_loader, ImageFolderWithPaths, \
    make_embeddings_path
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
import json
from config.goods_config import *
import scipy.io as sio

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        elif isinstance(obj,bytes):
            return str(obj,encoding='utf-8');
        return json.JSONEncoder.default(self,obj)


if __name__ == '__main__':

    model = EmbedNetworkResnet(1475)

    model_path = './model'



    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.Parameter = checkpoint['cls_dict']
    # torch.save(model, 'model.pth')
    if cuda:
        model = torch.nn.parallel.DataParallel(model)
        model.cuda()
    dataset_path = r'/home/sl/.shelf/classification/knn_models/register_classified_cropped_images'
    register_dataset = ImageFolderWithPaths(dataset_path, transform=dev_test_transform,
                                            loader=exampler_loader)

    register_loader = DataLoader(register_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    label_to_id = register_dataset.id_to_class

    # with open(label_to_id_path, 'w') as f:
    #     json.dump(label_to_id, f)

    embeddings, labels, paths, ids = make_embeddings_path(register_loader, model)
    embeddings = normalize(embeddings)
    print(embeddings.shape)
    with open('embeddings', 'w') as f:
        json.dump(embeddings,cls=MyEncoder,fp=f)
    with open('label_to_id', 'w') as f:
        json.dump(label_to_id,cls=MyEncoder,fp=f)