"""
Code for generating knn model
"""
import xlrd
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
import time
import pickle

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        elif isinstance(obj,bytes):
            return str(obj,encoding='utf-8');
        return json.JSONEncoder.default(self,obj)


if __name__ == '__main__':


    dataset_path = '/Data/open_shelf/'
    dataset_list_path = dataset_path
    # make_directories = True
    generate_dataset_list = False
    get_classified_images = False
    copy_register_images = False
    generate_label_to_name = True
    use_embeddings_to_generate_knn = False

    save_path = os.path.expanduser('~') + "/.shelf/classification/knn_models/" + time.strftime("%Y%m%d",
                                                                                               time.localtime()) + "/"
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(os.path.expanduser('~') + "/.shelf/classification/knn_models/" + time.strftime("%Y%m%d",
                                                                                                time.localtime()) + "/")

    register_classified_cropped_images_path = \
        os.path.expanduser('~') + '/.shelf/classification/knn_models/register_classified_cropped_images/'
    if not os.path.exists(register_classified_cropped_images_path):
        os.makedirs(register_classified_cropped_images_path)

    workbook1 = xlrd.open_workbook(r'/Data/open_shelf/open_shelf.xlsx')
    sheet1_name = workbook1.sheet_names()[0]
    sheet1 = workbook1.sheet_by_name(sheet1_name)

    workbook2 = xlrd.open_workbook(r'/home/sl/.shelf/classification/knn_models/20190311/open_shelf.xlsx')
    sheet2_name = workbook2.sheet_names()[1]
    sheet2 = workbook2.sheet_by_name(sheet2_name)


    # if make_directories:
    for row in range(sheet1.nrows):
        cur_path = dataset_path + 'classified_cropped_images/' + sheet1.cell_value(row, 0)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)

    if generate_dataset_list:
        dataset_list = get_whole_dataset_list(dataset_path)
        with open(os.path.join(dataset_list_path, 'dataset_list.pck'), 'wb') as p:
            pickle.dump(dataset_list, p)
    else:
        with open(os.path.join(dataset_list_path, 'dataset_list.pck'), 'rb') as p:
            dataset_list = pickle.load(p)

    if get_classified_images:
        total_image_num = len(dataset_list)
        each_class_image_num = np.zeros(sheet1.nrows, dtype=int)
        for _data_index in range(total_image_num):
            _data = dataset_list[_data_index]
            if (_data[0][3] + _data[0][1]) * 0.5 > 295:
                region = _data[0]
                name = _data[1]
                for row in range(sheet1.nrows):
                    if name == sheet1.cell_value(row, 0):
                        label = sheet1.cell_value(row, 0)
                        temp = each_class_image_num[row].__str__().zfill(7)
                        save_name = label + '_' + each_class_image_num[row].__str__().zfill(7) + '.jpg'
                        each_class_image_num[row] += 1

                        origin_image_path = _data[2]
                        save_image_path = dataset_path + 'classified_cropped_images/' + label + '/' + save_name
                        with open(origin_image_path, 'rb') as f:
                            img_origin = Image.open(f)
                            img_crop = img_origin.crop(region)
                            img_crop.save(save_image_path)
            if _data_index % 10000 == 0:
                print('[{}/{} ({:.0f}%)] '.format(_data_index, total_image_num, 100. * _data_index / total_image_num))

    if copy_register_images:
        origin_classified_cropped_image_path = dataset_path + 'classified_cropped_images/'
        class_list = os.listdir(origin_classified_cropped_image_path)
        for _class in class_list:
            for _row in range(sheet2.nrows):
                if _class == sheet2.cell_value(_row, 0):
                    shutil.copytree(origin_classified_cropped_image_path + _class, register_classified_cropped_images_path + _class)

    if generate_label_to_name:
        label_to_name = {}
        for row in range(sheet2.nrows):
            label_to_name[sheet2.cell_value(row, 0)] = sheet2.cell_value(row, 1)

        with open(os.path.expanduser('~') +"/.shelf/classification/knn_models/"+time.strftime("%Y%m%d",time.localtime())+"/label_to_name.json", 'w',encoding='utf-8') as p:
            json.dump(label_to_name,cls=MyEncoder,fp=p)

    print('finish!')

    if not use_embeddings_to_generate_knn:
        model = EmbedNetworkResnet(1229)

        model_path =os.path.expanduser('~') + '/.shelf/classification/models/20190313/epoch_55'



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

        with open(os.path.expanduser('~') +r"/.shelf/classification/knn_models/"+time.strftime("%Y%m%d",time.localtime())+"/register_embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings,file=f)
        with open(os.path.expanduser('~') +"/.shelf/classification/knn_models/"+time.strftime("%Y%m%d",time.localtime())+"/index_to_label.json", 'w') as f:
            json.dump(label_to_id,cls=MyEncoder,fp=f)
