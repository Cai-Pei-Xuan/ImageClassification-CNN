# -*-coding:UTF-8 -*-
import os
import glob
from random import shuffle
import shutil

# 分割資料集的參考網站:https://medium.com/@shihaoticking/%E5%AF%A6%E4%BD%9C%E8%B3%87%E6%96%99%E5%BC%B7%E5%8C%96-data-augmentation-%E5%AF%A6%E7%8F%BE%E5%9C%96%E7%89%87%E7%BF%BB%E8%BD%89-%E5%B9%B3%E7%A7%BB-%E7%B8%AE%E6%94%BE-4b37d4400ffb
def splitDataset(Origin_data_folder, class_list):
    for cl in class_list:
        img_path = os.path.join(Origin_data_folder, cl)                          # 取得單一類別資料夾路徑
        images = glob.glob(img_path + '/*.jpg')                        # 載入所有 jpg 檔成為一個 list
        shuffle(images)
        print("{}: {} Images".format(cl, len(images)))                 # 印出單一類別有幾張圖片
        num_train = int(round(len(images)*0.5))                        # 切割 50% 資料作為訓練集
        train_list, test_list = images[:num_train], images[num_train:]            

        for train in train_list:
            if not os.path.exists(os.path.join(Origin_data_folder, 'train', cl)):  # 如果資料夾不存在
                os.makedirs(os.path.join(Origin_data_folder, 'train', cl))           # 建立新資料夾
            shutil.move(train, os.path.join(Origin_data_folder, 'train', cl))          # 搬運圖片資料到新的資料夾

        for test in test_list:
            if not os.path.exists(os.path.join(Origin_data_folder, 'test', cl)):    # 如果資料夾不存在
                os.makedirs(os.path.join(Origin_data_folder, 'test', cl))             # 建立新資料夾
            shutil.move(test, os.path.join(Origin_data_folder, 'test', cl))            # 搬運圖片資料到新的資料夾


if __name__ == "__main__":
    Origin_data_folder = 'money'
    class_list = ['100','500','1000']
    splitDataset(Origin_data_folder, class_list)

