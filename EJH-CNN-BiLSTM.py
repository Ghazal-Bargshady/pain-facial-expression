import numpy as np
import keras
import tensorflow as tf
from keras.engine import  Model
from keras.layers import Lambda, Flatten, Dense, Input
from keras_preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import urllib.request
import urllib

from keras.optimizers import SGD
from keras_vggface.models import RESNET50, VGG16, SENET50
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import InputLayer, Embedding, TimeDistributed, Dropout, Bidirectional, Conv1D
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from sklearn.model_selection import KFold
import os
import cv2
import tflearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tflearn import bidirectional_rnn
from tflearn import  BasicLSTMCell
from tflearn.layers.core import dropout, fully_connected
from sklearn.model_selection import LeaveOneOut 
from sklearn.metrics import confusion_matrix
from keras_preprocessing import image
from keras_vggface import utils
import unittest
from sklearn.decomposition import PCA
from sklearn import decomposition
from keras_contrib.layers import CRF
from keras.layers import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
import itertools
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
from imblearn.over_sampling import SMOTE
from skimage import io, color
from skimage import exposure
import scipy
from keras.models import Sequential, Model, load_model
from keras import applications
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized


X=[]
Y=[]
X2=[]
Y2=[]
x0=0
x1=0
x2=0
x3=0
P_folder='D:/MyPhD/balancedata/PSPI/'
i_folder='D:/MyPhD/balancedata/Images/'
p_paitients=os.listdir(P_folder)

for k in range(len(p_paitients)):
    PSPI_folder='D:/MyPhD/balancedata/PSPI' + '/' + p_paitients[k]
    img_folder='D:/MyPhD/balancedata/Images' + '/' + p_paitients[k]
    folders=os.listdir(PSPI_folder)
       
    for i in range(len(folders)):
        foldername=folders[i]
        PSPI_folder1=os.path.join(PSPI_folder,foldername)
        img_folder1=os.path.join(img_folder,foldername)
        imgfiles=os.listdir(img_folder1)
        pspifiles=os.listdir(PSPI_folder1)
        
        for j in range(len(pspifiles)):
            file_name=pspifiles[j]      
            PSPIfile_path=os.path.join(PSPI_folder1,pspifiles[j])
            imgfile_path=os.path.join(img_folder1,imgfiles[j])
            PSPIfile = open(PSPIfile_path, "r")
            cs=int(float(str(PSPIfile.read()).replace('\n','')))
            if cs==0:
                cs=0
                x0=x0+1
            if cs==1:
                cs=1
                x1=x1+1
            if cs==2 or cs==3:
                cs=2
                x2=x2+1
            if cs==4 or cs==5 or cs==6 or cs==7 or cs==8 or cs==9 or cs==10 or cs==11 or cs==12 or cs==13 or cs==14 or cs==15:
                cs=3
                x3=x3+1
                        
            Y.append(cs)
            img=cv2.imread(imgfile_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #resize
            if(img.shape[0] > img.shape[1]):
                tile_size = (int(img.shape[1]*256/img.shape[0]),256)
            else:
                tile_size = (256, int(img.shape[0]*256/img.shape[1]))
            #centering
            img = centering_image(cv2.resize(img, dsize=tile_size))
            #out put 224*224px
            img = img[16:240, 16:240]
            img = img.reshape(224,224,3)
            X.append(img)

def get_y(Y):
    output=[]
    n_tags = np.max(Y) + 1
    for i in range(len(Y)):
        vv=[]
        for k in range(n_tags):
            if(k==Y[i]):
                vv.append(1)
            else:
                vv.append(0)
        output.append(vv)
    return output


X=np.array(X)
Y=np.array(Y)
n_tags=np.max(Y)+1
YY=np.array(get_y(Y))
print(n_tags)
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)
print(kf)

for train_index, test_index in kf.split(X):
    X_tr, X_te = X[train_index], X[test_index]
    y_tr, y_te = YY[train_index], YY[test_index]

    X_tr = X_tr.astype('float32')
    X_te = X_te.astype('float32')
    X_tr /= 255
    X_te /= 255

    vgg_model = VGGFace(include_top=False, input_shape=(224,224,3))
    print(vgg_model.summary())
    for layer in vgg_model.layers:
        layer.trainable = False
    xx = Flatten()(vgg_model.output)
    xx = Dense(1024, activation='relu')(xx)
    xx = Dropout(0.5)(xx)
    out = Dense(n_tags, activation='softmax')(xx)
    vgg_custom = Model(vgg_model.input, out)
    print(vgg_custom.summary())
    d1=X_tr.shape[0]
    print(d1)
    d2=X_te.shape[0]
    print(X_tr.shape)
    print(X_te.shape)
    vgg_custom.compile(loss='categorical_crossentropy', optimizer='adam',
             metrics=['accuracy'])
    vgg_custom.fit([X_tr], y_tr,  batch_size=48,
          epochs=1,
          validation_data=([X_te], y_te))
    features_train=vgg_custom.predict(X_tr)
    print (len(features_train))
    print (features_train)
    features_test=vgg_custom.predict(X_te)
    print(features_train.shape)
    print(features_test.shape)
    print('need change')
    features_train=features_train.reshape(d1,4)
    features_test=features_test.reshape(d2,4)
    pca = PCA(n_components=4)
    pca.fit(features_train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    X_pca_train = pca.fit_transform(features_train)
    X_pca_test = pca.transform(features_test)
    print('strat')
    pca_std = np.std(X_pca_train)
    print('finish')
    print(X_pca_train.shape)
    print(X_pca_test.shape)
    print(pca_std)
    print (len(X_pca_train))
    print (X_pca_train)
    input_lay = Input(shape=(1,4))
    x = Conv1D(256, 1, activation='relu')(input_lay)
    x = Conv1D(128, 1, activation='relu')(x)
    lstm_lay = Bidirectional(LSTM(256))(x)
    lstm_lay =Dense(4096)(lstm_lay)
    lstm_lay =Dropout(0.5)(lstm_lay)
    lstm_lay =GaussianNoise(pca_std)(lstm_lay)
    output_lay = Dense(n_tags, activation='relu')(lstm_lay)
    model1 = Model(inputs=[input_lay], outputs=[output_lay])
    model1.summary()
    lstm_lay2 = Bidirectional(LSTM(32))(x)
    lstm_lay2 =Dense(4096)(lstm_lay2)
    lstm_lay2 =Dropout(0.5)(lstm_lay2)
    lstm_lay=GaussianNoise(pca_std)(lstm_lay)
    output_lay2 = Dense(n_tags, activation='relu')(lstm_lay2)
    model2 = Model(inputs=[input_lay], outputs=[output_lay2])
    model2.summary()
    mergedOut = Add()([model1.output,model2.output])
    mergedOut=GaussianNoise(pca_std)(mergedOut)
    mergedOut = Dense(n_tags, activation='softmax')(mergedOut)
    model3 = Model(inputs=[model1.input], outputs=[mergedOut])
    model3.summary()
    X_pca_train=X_pca_train.reshape(d1,1,4)
    X_pca_test=X_pca_test.reshape(d2,1,4)
    model3.compile(loss='categorical_crossentropy', optimizer='adam',
          metrics=['mse', 'mae','accuracy'])
    model3.fit(X_pca_train, y_tr,  batch_size=48,
         epochs=1,
      validation_data=(X_pca_test, y_te))
    y_pred = model3.predict(X_pca_test)
    auc = roc_auc_score(y_te, y_pred)
    print('AUC: %.3f' % auc)
    cm = confusion_matrix(y_te.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure()
    plot_confusion_matrix(cm, classes=["no pain","weak pain","mild pain", "strong pain"])
    plt.show()
    K.clear_session()

