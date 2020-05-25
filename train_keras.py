'''
this is for classification metastasis or not
'''
import os
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import fire
from models_keras import SurvivalModel
import xlrd

def gen_dataset_with_mask_cut_thres_same_resolution(data_dir, dict_f):
    mask_dir = data_dir + '_mask'
    img_ori_dir = data_dir

    datasets = []
    datasets_name = []

    f_list = os.listdir(img_ori_dir)

    count = 0
    for index, i in enumerate(f_list):
        name, _, _ = os.path.splitext(i)[0].split('_')
        if name not in dict_f.keys():
            continue
        trans_num = dict_f[name]
        cls_y = int(trans_num>0)
        if cls_y>0:
            count+=1
    
        X = np.load(os.path.join(img_ori_dir, i))
        X = np.transpose(X,(1,2,0))


        m = np.load(os.path.join(mask_dir, i))
        X0 = X.copy()
        m0 = m.copy()
        #m = (m*255).astype(np.uint8).copy()
        #print (np.max(m), np.min(m), np.unique(m))
        #_, cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print (cnts)

        '''
        if data_dir.split('/')[-1] == 'train_single': 
            img_max = np.max(X[X<175])
            img_min = np.min(X[X>-75])
            X[X>175] = img_max
            X[X<-75] = img_min
        elif data_dir.split('/')[-1] == 'test_single': 
            img_max = np.max(X[X<125])
            img_min = np.min(X[X>-125])
            X[X>125] = img_max
            X[X<-125] = img_min
        '''

        #img_max = np.max(X[X<125])
        #img_min = np.min(X[X>-125])
        #X[X>125] = img_max
        #X[X<-125] = img_min

        #X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #X = clahe.apply(X)
        #X = (X-np.min(X))/(np.max(X)-np.min(X))
        
        #single_img_save_path = data_dir.rstrip('/') + '_img'
        #os.makedirs(single_img_save_path, exist_ok=True)
        #io.imsave(os.path.join(single_img_save_path, f'{name}_{index}.jpg'), X)

        '''
        if data_dir.split('/')[-1] == 'train_single': 
            train_max = max(train_max, X[m>0].max())
            train_min = min(train_min, X[m>0].min())
        elif data_dir.split('/')[-1] == 'test_single': 
            test_max = max(test_max, X[m>0].max())
            test_min = min(test_min, X[m>0].min())

        continue
        '''
        x, y = np.where(m>0)
        x_mean, y_mean = np.mean(x), np.mean(y)
        #print (x_mean, y_mean)
        w0, h0 = m.shape

        #x_min = max(0, int(np.min(x)-5))
        #x_max = min(w0, int(np.max(x)+5))
        #y_min = max(0, int(np.min(y)-5))
        #y_max = min(h0, int(np.max(y)+5))

        x_min = max(0, int(x_mean-80))
        x_max = min(w0, int(x_mean+80))
        y_min = max(0, int(y_mean-80))
        y_max = min(h0, int(y_mean+80))



        #print (x_min, x_max, y_min, y_max)
        m = m[x_min:x_max, y_min:y_max]
        X = X[x_min:x_max, y_min:y_max, :]

        #X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #X = clahe.apply(X)

        X_m_1 = X.copy() 
        #X_m_1[m<1.0] = 0


        #X_m_1 = ((X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0])))*0.9+0.1
        X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        X = (X-np.min(X))/(np.max(X)-np.min(X))
        X_m_1[m==0] = 0
        #print (np.unique(X_m_1))
        #raise


        X_m_2 = X.copy() 
        X_m_2[m>0] = 0


        #print (X_m_1.max(), X_m_1.min(), np.unique(X_m_1))
        #print (X_m_1[X_m_1>0].max(), X_m_1[X_m_1>0].min(), np.unique(X_m_1[X_m_1>0]))
        #plt.imshow(X_m_1)
        #plt.show()
        #raise        

        h, w, _ = X_m_1.shape
        #print (w, h)

        if h < w:
            pad_1 = (w - h)//2
            pad_2 = w - pad_1 - h
            X_m_1 = np.lib.pad(X_m_1, ((pad_1, pad_2),(0,0),(0,0)), 'constant', constant_values=(0,0))
            X = np.lib.pad(X, ((pad_1, pad_2),(0,0),(0,0)), 'constant', constant_values=(0,0))
            m = np.lib.pad(m, ((pad_1, pad_2),(0,0),(0,0)), 'constant', constant_values=(0,0))
        elif h >= w:
            pad_1 = (h - w)//2
            pad_2 = h - pad_1 - w
            X_m_1 = np.lib.pad(X_m_1, ((0, 0),(pad_1, pad_2),(0,0)), 'constant', constant_values=(0,0))
            X = np.lib.pad(X, ((0, 0),(pad_1, pad_2),(0,0)), 'constant', constant_values=(0,0))
            m = np.lib.pad(m, ((0, 0),(pad_1, pad_2)), 'constant', constant_values=(0,0))

        #print (X_m_1.min(), X_m_1.max())

        if X_m_1.shape[0] != 160 or X_m_1.shape[1] != 160:
            #X = cv2.resize(X, (96, 96), interpolation=cv2.INTER_CUBIC)
            #m = cv2.resize(m, (96, 96), interpolation=cv2.INTER_CUBIC)
            #X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_NEAREST)
            X = cv2.resize(X, (160, 160), interpolation=cv2.INTER_CUBIC)
            X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_CUBIC)
            m = cv2.resize(m, (160, 160), interpolation=cv2.INTER_CUBIC)
            #X_m_2 = cv2.resize(X_m_2, (96, 96), interpolation=cv2.INTER_CUBIC)

        #X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        #X_m_1[m==0] = 0

        #print (X_m_1.max(), X_m_1.min(), X_m_1.shape)
        #print (m.max(), m.min(), m.shape)
        #raise

        if m0.shape[0] != 160 or m0.shape[1] != 160:
            m0 = cv2.resize(m0, (160, 160), interpolation=cv2.INTER_CUBIC)

        #print (X_m_1.min(), X_m_1.max())
        #raise

        X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        X = (X-np.min(X))/(np.max(X)-np.min(X))
        m = (m-np.min(m))/(np.max(m)-np.min(m))
        X_m_1[m<=0] = 0
        #print (X.shape, np.max(X_m_1), np.min(X_m_1))
        #raise

        '''
        single_img_save_path = data_dir.rstrip('/') + '_img_cut'
        os.makedirs(single_img_save_path, exist_ok=True)
        io.imsave(os.path.join(single_img_save_path, f'{name}_{cls_y}.jpg'), X_m_1)

        single_img_save_path_X = data_dir.rstrip('/') + '_img_cut_X'
        os.makedirs(single_img_save_path_X, exist_ok=True)
        io.imsave(os.path.join(single_img_save_path_X, f'{name}_{cls_y}.jpg'), X)

        single_img_save_path_m = data_dir.rstrip('/') + '_img_cut_m'
        os.makedirs(single_img_save_path_m, exist_ok=True)
        io.imsave(os.path.join(single_img_save_path_m, f'{name}_{cls_y}.jpg'), m)
        '''

        #X_m_1 = np.expand_dims(X_m_1, axis=2)
        #X = np.expand_dims(X, axis=2)
        m = np.expand_dims(m, axis=2)
        m0 = np.expand_dims(m0, axis=2)

        #X_m_2 = np.expand_dims(X_m_2, axis=2)

        #XX = np.concatenate((X_m_1, X_m_1, X_m_1), axis=-1) 
        #XX = np.concatenate((m0, m0, m0), axis=-1) 
        XX = np.concatenate((X, m), axis=-1) 
        #XX = np.concatenate((m, m, m), axis=-1) 
        #XX = X_m_1
        #XX = m 

        Y = np.array([cls_y, 1-cls_y])
        #Y = np.array([1-cls_y, cls_y])

        datasets.append((XX[None,...], Y[None,...]))
        datasets_name.append(name)
        #print (name, cls_y)
        #print (float(os2), int(event))
        #print (X.shape, np.max(X), np.min(X))

    #print (data_dir)
    #print (train_max, train_min, test_max, test_min)
    #if data_dir.split('/')[-1] == 'train_single': 
    set_name = data_dir.split('/')[-1]
    print (f'{set_name}: {len(datasets)}')
    print (f'{set_name}: {len(datasets_name)}')
    print ('count:', count)
    #raise
    return datasets, datasets_name



def gen_dataset_with_mask_cut_thres(data_dir, dict_f):
    mask_dir = data_dir + '_mask'
    img_ori_dir = data_dir + '_ori'

    datasets = []
    datasets_name = []

    f_list = os.listdir(img_ori_dir)

    count = 0
    for index, i in enumerate(f_list):
        name, _, _ = os.path.splitext(i)[0].split('_')
        if name not in dict_f.keys():
            continue
        trans_num = dict_f[name]
        cls_y = int(trans_num>0)
        if cls_y>0:
            count+=1
    
        X = np.load(os.path.join(img_ori_dir, i))
        m = np.load(os.path.join(mask_dir, i))
        X0 = X.copy()
        m0 = m.copy()
        #m = (m*255).astype(np.uint8).copy()
        #print (np.max(m), np.min(m), np.unique(m))
        #_, cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print (cnts)

        '''
        if data_dir.split('/')[-1] == 'train_single': 
            img_max = np.max(X[X<175])
            img_min = np.min(X[X>-75])
            X[X>175] = img_max
            X[X<-75] = img_min
        elif data_dir.split('/')[-1] == 'test_single': 
            img_max = np.max(X[X<125])
            img_min = np.min(X[X>-125])
            X[X>125] = img_max
            X[X<-125] = img_min
        '''

        #img_max = np.max(X[X<125])
        #img_min = np.min(X[X>-125])
        #X[X>125] = img_max
        #X[X<-125] = img_min

        #X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #X = clahe.apply(X)
        #X = (X-np.min(X))/(np.max(X)-np.min(X))
        
        #single_img_save_path = data_dir.rstrip('/') + '_img'
        #os.makedirs(single_img_save_path, exist_ok=True)
        #io.imsave(os.path.join(single_img_save_path, f'{name}_{index}.jpg'), X)

        '''
        if data_dir.split('/')[-1] == 'train_single': 
            train_max = max(train_max, X[m>0].max())
            train_min = min(train_min, X[m>0].min())
        elif data_dir.split('/')[-1] == 'test_single': 
            test_max = max(test_max, X[m>0].max())
            test_min = min(test_min, X[m>0].min())

        continue
        '''
        x, y = np.where(m>0)

        #x_mean, y_mean = np.mean(x), np.mean(y)
        #print (x_mean, y_mean)
        w0, h0 = m.shape
        x_min = max(0, int(np.min(x)-5))
        x_max = min(w0, int(np.max(x)+5))
        y_min = max(0, int(np.min(y)-5))
        y_max = min(h0, int(np.max(y)+5))

        #print (x_min, x_max, y_min, y_max)
        m = m[x_min:x_max, y_min:y_max]
        X = X[x_min:x_max, y_min:y_max] 

        X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        X = clahe.apply(X)

        X_m_1 = X.copy() 
        #X_m_1[m<1.0] = 0


        #X_m_1 = ((X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0])))*0.9+0.1
        X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        X = (X-np.min(X))/(np.max(X)-np.min(X))
        X_m_1[m==0] = 0
        #print (np.unique(X_m_1))
        #raise


        X_m_2 = X.copy() 
        X_m_2[m>0] = 0


        #print (X_m_1.max(), X_m_1.min(), np.unique(X_m_1))
        #print (X_m_1[X_m_1>0].max(), X_m_1[X_m_1>0].min(), np.unique(X_m_1[X_m_1>0]))
        #plt.imshow(X_m_1)
        #plt.show()
        #raise        

        h, w = X_m_1.shape
        #print (w, h)

        if h < w:
            pad_1 = (w - h)//2
            pad_2 = w - pad_1 - h
            X_m_1 = np.lib.pad(X_m_1, ((pad_1, pad_2),(0,0)), 'constant', constant_values=(0, 0))
            X = np.lib.pad(X, ((pad_1, pad_2),(0,0)), 'constant', constant_values=(0, 0))
            m = np.lib.pad(m, ((pad_1, pad_2),(0,0)), 'constant', constant_values=(0, 0))
        elif h >= w:
            pad_1 = (h - w)//2
            pad_2 = h - pad_1 - w
            X_m_1 = np.lib.pad(X_m_1, ((0, 0),(pad_1, pad_2)), 'constant', constant_values=(0, 0))
            X = np.lib.pad(X, ((0, 0),(pad_1, pad_2)), 'constant', constant_values=(0, 0))
            m = np.lib.pad(m, ((0, 0),(pad_1, pad_2)), 'constant', constant_values=(0, 0))

        #print (X_m_1.min(), X_m_1.max())

        if X_m_1.shape[0] != 160 or X_m_1.shape[1] != 160:
            #X = cv2.resize(X, (96, 96), interpolation=cv2.INTER_CUBIC)
            #m = cv2.resize(m, (96, 96), interpolation=cv2.INTER_CUBIC)
            #X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_NEAREST)
            X = cv2.resize(X, (160, 160), interpolation=cv2.INTER_CUBIC)
            X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_CUBIC)
            m = cv2.resize(m, (160, 160), interpolation=cv2.INTER_CUBIC)
            #X_m_2 = cv2.resize(X_m_2, (96, 96), interpolation=cv2.INTER_CUBIC)

        #X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        #X_m_1[m==0] = 0

        #print (X_m_1.max(), X_m_1.min(), X_m_1.shape)
        #print (m.max(), m.min(), m.shape)
        #raise

        if m0.shape[0] != 160 or m0.shape[1] != 160:
            m0 = cv2.resize(m0, (160, 160), interpolation=cv2.INTER_CUBIC)

        #print (X_m_1.min(), X_m_1.max())
        #raise

        X_m_1 = (X_m_1-np.min(X_m_1[m>0]))/(np.max(X_m_1[m>0])-np.min(X_m_1[m>0]))
        X = (X-np.min(X))/(np.max(X)-np.min(X))
        m = (m-np.min(m))/(np.max(m)-np.min(m))
        X_m_1[m<=0] = 0
        #print (X.shape, np.max(X_m_1), np.min(X_m_1))
        #raise

        single_img_save_path = data_dir.rstrip('/') + '_img_cut'
        os.makedirs(single_img_save_path, exist_ok=True)
        io.imsave(os.path.join(single_img_save_path, f'{name}_{cls_y}.jpg'), X_m_1)

        single_img_save_path_X = data_dir.rstrip('/') + '_img_cut_X'
        os.makedirs(single_img_save_path_X, exist_ok=True)
        io.imsave(os.path.join(single_img_save_path_X, f'{name}_{cls_y}.jpg'), X)

        single_img_save_path_m = data_dir.rstrip('/') + '_img_cut_m'
        os.makedirs(single_img_save_path_m, exist_ok=True)
        io.imsave(os.path.join(single_img_save_path_m, f'{name}_{cls_y}.jpg'), m)

        X_m_1 = np.expand_dims(X_m_1, axis=2)
        X = np.expand_dims(X, axis=2)
        m = np.expand_dims(m, axis=2)
        m0 = np.expand_dims(m0, axis=2)

        #X_m_2 = np.expand_dims(X_m_2, axis=2)

        #XX = np.concatenate((X_m_1, X_m_1, X_m_1), axis=-1) 
        #XX = np.concatenate((m0, m0, m0), axis=-1) 
        XX = np.concatenate((X, m, X), axis=-1) 
        #XX = np.concatenate((m, m, m), axis=-1) 
        #XX = X_m_1
        #XX = m 

        Y = np.array([cls_y, 1-cls_y])
        #Y = np.array([1-cls_y, cls_y])

        datasets.append((XX[None,...], Y[None,...]))
        datasets_name.append(name)
        #print (name, cls_y)
        #print (float(os2), int(event))
        #print (X.shape, np.max(X), np.min(X))

    #print (data_dir)
    #print (train_max, train_min, test_max, test_min)
    #if data_dir.split('/')[-1] == 'train_single': 
    set_name = data_dir.split('/')[-1]
    print (f'{set_name}: {len(datasets)}')
    print (f'{set_name}: {len(datasets_name)}')
    print ('count:', count)
    #raise
    return datasets, datasets_name


def dataset_balance(datasets, name):
    print (len(datasets))
    print (datasets[0][1][0])
    count_y = 0
    for i in range(len(datasets)):
        if datasets[i][1][0][0] > 0:
            count_y += 1
    print (count_y)
    raise


def load_dataset(data_path, dict_1200, dict_500):

    train_val_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    datasets_train_val, train_val_name = gen_dataset_with_mask_cut_thres_same_resolution(train_val_dir, dict_1200)
    datasets_test, test_name = gen_dataset_with_mask_cut_thres_same_resolution(test_dir, dict_500)

    #train_val_dir = os.path.join(data_path, 'train_single')
    #test_dir = os.path.join(data_path, 'test_val_single')
    #datasets_train_val, train_val_name = gen_dataset_with_mask_cut_thres(train_val_dir, dict_1200)
    #datasets_test, test_name = gen_dataset_with_mask_cut_thres(test_dir, dict_500)


    #raise
    split_n = int(len(datasets_train_val)/3*2)
    datasets_train, train_name = datasets_train_val[:split_n], train_val_name[:split_n]
    datasets_val, val_name = datasets_train_val[split_n:], train_val_name[split_n:]

    #datasets_train, train_name = dataset_balance(datasets_train, train_name)

    print ('train set len: ', len(datasets_train))
    print ('val set len: ', len(datasets_val))
    print ('test set len: ', len(datasets_test))

    return datasets_train, train_name, datasets_val, val_name, datasets_test, test_name


def main(data_path='data_multilayer_new', excel_1200_path='SYU_1200.xlsx', excel_500_path='south_hospital_700.xlsx', excel_500_miss_path='name_500_excel_miss.xlsx'):

    excel_1200 = xlrd.open_workbook(excel_1200_path)
    excel_500 = xlrd.open_workbook(excel_500_path)
    excel_500_miss = xlrd.open_workbook(excel_500_miss_path)

    sheet_1200 = excel_1200.sheet_by_index(0)
    sheet_500 = excel_500.sheet_by_index(0)
    sheet_500_miss = excel_500_miss.sheet_by_index(0)

    dict_1200 = {}
    dict_500 = {}
    dict_500_miss = {}

    #print (sheet_500_miss)
    #raise

    select_index = 31

    if select_index > 7:
        offset_1200 = 4
    else:
        offset_1200 = 2

    if select_index == 7:
        print (sheet_1200.row_values(1)[select_index+offset_1200])
        print (sheet_500.row_values(0)[select_index])
        print (sheet_500_miss.row_values(0)[select_index])

        print (sheet_500.row_values(0)[select_index+2])
        print (sheet_500_miss.row_values(0)[select_index+2])

        print (sheet_500.row_values(0)[select_index+4])
        print (sheet_500_miss.row_values(0)[select_index+4])

        for row in range(2, sheet_1200.nrows):
            if str(sheet_1200.row_values(row)[select_index+offset_1200]).replace(' ',''):
                name = sheet_1200.row_values(row)[0]
                if name.replace(' ', '')=='黄荣贵':
                    continue
                dict_1200[name.replace(' ','')] = int(sheet_1200.row_values(row)[select_index+offset_1200])

        for row in range(1, sheet_500.nrows):
            if sheet_500.row_values(row)[select_index] or sheet_500.row_values(row)[select_index+2] or sheet_500.row_values(row)[select_index+4]:
                name = sheet_500.row_values(row)[0]
                dict_500[name.replace(' ','')] = 0
            if sheet_500.row_values(row)[select_index]:
                dict_500[name.replace(' ','')] = dict_500[name.replace(' ','')] + int(sheet_500.row_values(row)[select_index])
            if sheet_500.row_values(row)[select_index+2]:
                dict_500[name.replace(' ','')] = dict_500[name.replace(' ','')] + int(sheet_500.row_values(row)[select_index+2])
            if sheet_500.row_values(row)[select_index+4]:
                dict_500[name.replace(' ','')] = dict_500[name.replace(' ','')] + int(sheet_500.row_values(row)[select_index+4])

        for row in range(1, sheet_500_miss.nrows):
            if sheet_500_miss.row_values(row)[select_index] or sheet_500_miss.row_values(row)[select_index+2] or sheet_500_miss.row_values(row)[select_index+4]:
                name = sheet_500_miss.row_values(row)[0]
                dict_500_miss[name.replace(' ','')] = 0
            if sheet_500_miss.row_values(row)[select_index]:
                dict_500_miss[name.replace(' ','')] = dict_500_miss[name.replace(' ','')] + int(sheet_500_miss.row_values(row)[select_index])
            if sheet_500_miss.row_values(row)[select_index+2]:
                dict_500_miss[name.replace(' ','')] = dict_500_miss[name.replace(' ','')] + int(sheet_500_miss.row_values(row)[select_index+2])
            if sheet_500_miss.row_values(row)[select_index+4]:
                dict_500_miss[name.replace(' ','')] = dict_500_miss[name.replace(' ','')] + int(sheet_500_miss.row_values(row)[select_index+4])


    elif select_index == 270:
        print (sheet_1200.row_values(1)[select_index+offset_1200])
        print (sheet_500.row_values(0)[select_index])
        print (sheet_500_miss.row_values(0)[select_index])
        print (sheet_1200.row_values(1)[select_index+offset_1200+2])
        print (sheet_500.row_values(0)[select_index+2])
        print (sheet_500_miss.row_values(0)[select_index+2])
        #raise

        for row in range(2, sheet_1200.nrows):
            if str(sheet_1200.row_values(row)[select_index+offset_1200]).replace(' ','') or str(sheet_1200.row_values(row)[select_index+offset_1200+2]).replace(' ',''):
                name = sheet_1200.row_values(row)[0]
                if name.replace(' ', '')=='黄荣贵':
                    continue

                dict_1200[name.replace(' ','')] = 0
                if str(sheet_1200.row_values(row)[select_index+offset_1200]).replace(' ',''):
                    dict_1200[name.replace(' ','')] = dict_1200[name.replace(' ','')] + int(sheet_1200.row_values(row)[select_index+offset_1200])
                if str(sheet_1200.row_values(row)[select_index+offset_1200+2]).replace(' ',''):
                    dict_1200[name.replace(' ','')] = dict_1200[name.replace(' ','')] + int(sheet_1200.row_values(row)[select_index+offset_1200+2])

        for row in range(1, sheet_500.nrows):
            if sheet_500.row_values(row)[select_index] or sheet_500.row_values(row)[select_index+2]:
                name = sheet_500.row_values(row)[0]
                dict_500[name.replace(' ','')] = 0
            if sheet_500.row_values(row)[select_index]:
                dict_500[name.replace(' ','')] = dict_500[name.replace(' ','')] + int(sheet_500.row_values(row)[select_index])
            if sheet_500.row_values(row)[select_index+2]:
                dict_500[name.replace(' ','')] = dict_500[name.replace(' ','')] + int(sheet_500.row_values(row)[select_index+2])

        for row in range(1, sheet_500_miss.nrows):
            if sheet_500_miss.row_values(row)[select_index] or sheet_500_miss.row_values(row)[select_index+2]:
                name = sheet_500_miss.row_values(row)[0]
                dict_500_miss[name.replace(' ','')] = 0
            if sheet_500_miss.row_values(row)[select_index]:
                dict_500_miss[name.replace(' ','')] = dict_500_miss[name.replace(' ','')] + int(sheet_500_miss.row_values(row)[select_index])
            if sheet_500_miss.row_values(row)[select_index+2]:
                dict_500_miss[name.replace(' ','')] = dict_500_miss[name.replace(' ','')] + int(sheet_500_miss.row_values(row)[select_index+2])

    else:
        print (sheet_1200.row_values(1)[select_index+offset_1200])
        print (sheet_500.row_values(0)[select_index])
        print (sheet_500_miss.row_values(0)[select_index])

        for row in range(2, sheet_1200.nrows):
            if str(sheet_1200.row_values(row)[select_index+offset_1200]).replace(' ',''):
                name = sheet_1200.row_values(row)[0]
                if name.replace(' ', '')=='黄荣贵':
                    continue
                dict_1200[name.replace(' ','')] = int(sheet_1200.row_values(row)[select_index+offset_1200])

        for row in range(1, sheet_500.nrows):
            if sheet_500.row_values(row)[select_index]:
                name = sheet_500.row_values(row)[0]
                dict_500[name.replace(' ','')] = int(sheet_500.row_values(row)[select_index])

        for row in range(1, sheet_500_miss.nrows):
            if sheet_500_miss.row_values(row)[select_index]:
                name = sheet_500_miss.row_values(row)[0]
                dict_500_miss[name.replace(' ','')] = int(sheet_500_miss.row_values(row)[select_index])

    print (f'len 1200 excel {select_index+2} col:', len(dict_1200))
    print (f'len 500 excel {select_index} col:', len(dict_500))    
    print (f'len 500 excel miss {select_index} col:', len(dict_500_miss))   
    dict_500.update(dict_500_miss)
    print (f'len 500 excel {select_index} col after merge:', len(dict_500)) 

    #datasets_train, datasets_val, datasets_test, datasets_train_1, datasets_val_1, datasets_test_1 = load_dataset_avg(data_path)

    datasets_train, train_name, datasets_val, val_name, datasets_test, test_name = load_dataset(data_path, dict_1200, dict_500)

    #print (test_name)
    #print (len(datasets_train))

    #raise
    survival_model = SurvivalModel()

    #survival_model.run(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, epochs=1000, lr=0.01, mode='train', batch_size = 16)

    survival_model.run(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, epochs=1000, lr=0.01, mode='infer', batch_size = 16)

    #survival_model.run(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, epochs=1000, lr=0.01, mode='pred', batch_size = 16)

    #survival_model.run(datasets_train, datasets_val, datasets_test, train_name, val_name, test_name, epochs=1000, lr=0.01, mode='vis_cam', batch_size = 16)

if __name__ == '__main__':
    fire.Fire(main)


