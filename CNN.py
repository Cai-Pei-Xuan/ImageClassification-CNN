# -*-coding:UTF-8 -*-
# Convolutional Neural Networks(CNN)
import os
import matplotlib.pyplot as plt
# CNN模型架構參考網站:
# https://ithelp.ithome.com.tw/articles/10205389
# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/656352/
# https://www.jianshu.com/p/86d667ee3c62
# CNN stride和padding的細節:https://medium.com/@chih.sheng.huang821/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-convolutional-neural-network-cnn-%E5%8D%B7%E7%A9%8D%E8%A8%88%E7%AE%97%E4%B8%AD%E7%9A%84%E6%AD%A5%E4%BC%90-stride-%E5%92%8C%E5%A1%AB%E5%85%85-padding-94449e638e82
# Import Keras libraries and packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential         #用來啟動 NN
from keras.layers import Conv2D             # Convolution Operation
from keras.layers import MaxPooling2D       # Pooling
from keras.layers import SpatialDropout2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense              # Fully Connected Networks
# keras optimizers的用法:https://keras.io/zh/optimizers/
from keras import optimizers

# 建立一般的CNN模型
def makeCommonCNNModel(kernel_size, img_height, img_width, RGB, kind):
    model = Sequential([
        Conv2D(16, kernel_size, padding='SAME', activation='relu', input_shape=(img_height, img_width, RGB)),
        MaxPooling2D(2, 2),
        Conv2D(8, kernel_size, padding='SAME', activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(3, activation='softmax')
    ])
    Adam = optimizers.Adam(lr=5e-4)
    model.compile(optimizer=Adam,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
    return model

# 建立使用techniques的CNN模型，使用了data argumentation、regularization -weight decay，以及dropout來提高accuracy
def makeTechniquesCNNModel(kernel_size, img_height, img_width, RGB, kind):
    model = Sequential([
        Conv2D(16, kernel_size, padding='SAME', activation='relu', input_shape=(img_height, img_width, RGB)),
        MaxPooling2D(2, 2),
        SpatialDropout2D(0.2),
        Conv2D(8, kernel_size, padding='SAME', activation='relu'),
        MaxPooling2D(2, 2),
        SpatialDropout2D(0.2),
        Flatten(),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    Adam = optimizers.Adam(lr=5e-4, decay=0.001)
    model.compile(optimizer=Adam,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
    return model

# 繪畫正確率與loss，參考網站:http://sofasofa.io/forum_main_post.php?postid=1005503
def drawAccuracyAndLoss(history):
    epochs = len(history.history['loss'])
    plt.title('Train and Test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(range(epochs), history.history['accuracy'], label='accuracy')
    plt.plot(range(epochs), history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

    plt.figure()

    plt.title('Train and Test Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

# 使用一般的方式做訓練
def useCommonModel(batch_size, train_Path, test_Path, kernel_size, img_height, img_width, RGB, kind, epochs):
    image_gen_train = ImageDataGenerator(rescale=1./255)            # 從0~255整數，壓縮為0~1浮點數
    
    image_gen_val = ImageDataGenerator(rescale=1./255)              # 從0~255整數，壓縮為0~1浮點數

    train_data_gen = image_gen_train.flow_from_directory(
        batch_size=batch_size,
        directory=train_Path,
        shuffle=True,
        target_size=(img_height, img_width),
        class_mode='sparse'           # 分類標籤定義為 0, 1, 2
    )

    val_data_gen = image_gen_val.flow_from_directory(
        batch_size=batch_size,
        directory=test_Path,
        target_size=(img_height, img_width),
        class_mode='sparse'           # 分類標籤定義為 0, 1, 2
    )

    # 建立一般的CNN模型
    model = makeCommonCNNModel(kernel_size, img_height, img_width, RGB, kind)

    history = model.fit(
        train_data_gen,               # 帶入訓練資料產生器
        epochs=epochs,                # 將所有資料看過 50 次
        validation_data=val_data_gen  # 帶入驗證資料產生器
    )

    drawAccuracyAndLoss(history)      # 繪畫正確率與loss

# 使用多項技術(data argumentation、regularization -weight decay，以及dropout)做訓練
def useTechniquesModel(batch_size, train_Path, test_Path, kernel_size, img_height, img_width, RGB, kind, epochs):
    # 將train_data做增量
    image_gen_train = ImageDataGenerator(
        rescale=1./255,               # 從0~255整數，壓縮為0~1浮點數
        rotation_range=50,            # 隨機旋轉 ±45°
        width_shift_range=.15,        # 隨機水平移動 ±15%
    )
    
    image_gen_val = ImageDataGenerator(rescale=1./255)              # 從0~255整數，壓縮為0~1浮點數

    train_data_gen = image_gen_train.flow_from_directory(
        batch_size=batch_size,
        directory=train_Path,
        shuffle=True,
        target_size=(img_height, img_width),
        class_mode='sparse'           # 分類標籤定義為 0, 1, 2
    )

    val_data_gen = image_gen_val.flow_from_directory(
        batch_size=batch_size,
        directory=test_Path,
        target_size=(img_height, img_width),
        class_mode='sparse'           # 分類標籤定義為 0, 1, 2
    )

    # 建立使用techniques的CNN模型
    model = makeTechniquesCNNModel(kernel_size, img_height, img_width, RGB, kind)

    history = model.fit(
        train_data_gen,               # 帶入訓練資料產生器
        epochs=epochs,                # 將所有資料看過 50 次
        validation_data=val_data_gen  # 帶入驗證資料產生器
    )

    drawAccuracyAndLoss(history)      # 繪畫正確率與loss

if __name__ == "__main__":
    train_Path = 'money/train'
    test_Path = 'money/test'

    # 自訂參數
    batch_size = 8
    kernel_size = 5
    img_height = 128
    img_width = 128
    RGB = 3
    kind = 3
    epochs = 50
    
    # 使用一般的方式做訓練
    useCommonModel(batch_size, train_Path, test_Path, kernel_size, img_height, img_width, RGB, kind, epochs)
    # 使用多項技術(data argumentation、regularization -weight decay，以及dropout)做訓練
    useTechniquesModel(batch_size, train_Path, test_Path, kernel_size, img_height, img_width, RGB, kind, epochs)
