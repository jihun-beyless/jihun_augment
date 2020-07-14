import sys
import os
import augment
import json
import cv2
import numpy as np

sys.path.insert(0,'./DCD_DB_API-master/db_api/')
import DB

import time
#from db_api import DB

def img_loader(img_dir):
    if isinstance(img_dir, str):
        with open(img_dir, 'rb') as file:
            img = file.read()
    return img


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print("Failed to create director!!"+directory)

def tmp_image_save(DB_imgs, obj_id, g, g_id, obj_iter, img_size):    
    '''
    DB에서 받은 이미지를 tmp에 저장
    입력받은 category_id별로 이미지를 생성하며, 
    '''
    #폴더 생성
    folder_name = str('/tmp/augment_DB/{}/').format(obj_id)
    createFolder(folder_name)
    for x in range(1,grid[0]+1):
        for y in range(1,grid[1]+1):
            for iter_num in range(1,obj_iter+1):
                file_name = str('{}x{}_{}.jpg').format(x, y, iter_num)
                for img in DB_imgs:
                    if img[0:3]==(x,y,iter_num):
                        img_bytes = img[3]
                        img_np = np.frombuffer(img_bytes, dtype = np.uint8)
                        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        
                        #img = img_np.reshape((1080,1920,3))
                        #cv2.imshow('DB_images', img)
                        #cv2.waitKey(0)
                        file_path = folder_name+file_name
                        cv2.imwrite(file_path, img)

def arrange_masks(DB_masks, grid, obj_iter):
    #grid : 10
    #category :15, 16, 17
    #np_masks = np.array(masks)
    masks_list = list([[[None for iter in range(obj_iter)] for row in range(grid[1])] for col in range(grid[0])])
    for x in range(1,grid[0]+1):
         for y in range(1,grid[1]+1):
             for iter_num in range(1,obj_iter+1):
                mask_value = [list(obj_mask[3:6]) for obj_mask in DB_masks if (obj_mask[0:3]==(x,y,iter_num))]
                sort_mask = sorted(mask_value, key=lambda mask_value: mask_value[0])
                masks_list[x-1][y-1][iter_num-1] = [obj_mask[1:3] for obj_mask in sort_mask]
    return masks_list



def get_DB_data(db, obj_category, grid, grid_id, iteration, image_size):
    createFolder('/tmp/augment_DB')

    # DB접속
    #db = DB.DB('192.168.10.69', 3306, 'root', 'return123', 'test')

    masks = []
    for cate_id, obj_iter in zip(obj_category, iteration):
        DB_images = db.get_aug_img(str(grid_id), str(cate_id))
        tmp_image_save(DB_images, cate_id, grid, grid_id, obj_iter, image_size)
        DB_masks = db.get_aug_mask(str(grid_id), str(cate_id))
        masks_list = arrange_masks(DB_masks, grid,  obj_iter)
        masks.append(masks_list)

    return masks

def load_aug_images(images_path):
    images = []
    for path in images_path:
        img = cv2.imread(path)
        images.append(img)
    return images

def set_aug_result(db, aug_segs, grid, grid_id, device_id, images_path, obj_iter_start_num):
    obj_data_list = []
    bbox_data_list = []
    bbox_info = []
    
    #aug_images = load_aug_images(images_path)

    loc_id_tuple = db.get_aug_loc_id(grid_id)
    loc_id_table = list([[None for row in range(grid[1])] for col in range(grid[0])])

    for loc in loc_id_tuple:
        loc_id_table[loc[0]-1][loc[1]-1] = loc[2] 

    image_data_list = [[str(device_id), img_loader(img_p), '3', '1'] for img_p in images_path]

    start_img_id = db.get_last_id('Image')+1
    db.set_bulk_img(datas=image_data_list)
    end_img_id = db.get_last_id('Image')+1
    #end_time = time.time()
    #print('total_time: ', end_time - start_time)

    iter_num = obj_iter_start_num
    for img_segs, img_id  in zip(aug_segs, range(start_img_id, end_img_id)):
        #우선 obj 정보부터 저장
        for seg in img_segs:
            loc_id = loc_id_table[seg['x']][seg['y']]
            cate_id = seg['category_id']
            obj_data_list.append((str(img_id), str(loc_id), str(cate_id), str(iter_num), str(-1)))
            bbox_info.append(seg['bbox'])
            iter_num +=1

    print(obj_data_list)

    start_obj_id = db.get_last_id('Object')+1
    db.set_bulk_obj(datas=obj_data_list)
    end_obj_id = db.get_last_id('Object')+1

    for bbox, obj_id in zip(bbox_info, range(start_obj_id, end_obj_id)):
        bbox_data_list.append((str(obj_id), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])))

    print(bbox_data_list)
    db.set_bulk_bbox(datas=bbox_data_list)

    return iter_num



if __name__ == "__main__":
    # 이건 tool에서 입력으로 받아와야 하는 변수들
    device_id = 20017
    grid = (2,3)
    grid_id = 10
    object_category=[19, 21, 22]
    background = cv2.imread("background.jpg")
    iteration = 2
    batch1_num = 3000
    batch2_num = 3000
    batch3_num = 3000
    f = open("iter_num.txt", 'r')
    obj_iter_start_num = int(f.readline())
    f.close()
    f = open("iter_num.txt", 'w')
    f.write(str(20))
    f.close()

    if str(type(iteration))=="<class 'int'>" :
        iteration_list = [iteration for a in range(len(object_category))]
    else:
        iteration_list = iteration
    image_size = background.shape

    # DB접속
    db = DB.DB(ip='192.168.10.69', port=3306, user='root', password='return123', db_name='test')

    # 먼저 DB에서 file을 읽어오기
    DB_mask = get_DB_data(db, object_category, grid, grid_id, iteration_list, image_size)
    #DB_mask = get_data(object_category, grid, grid_id, iteration_list, image_size)


    #조건에 따라서 몇장 만들지 
    result_data = []
    aug_count = 1
    cut_value = 5
    img_path_list = []
    batch_method_list = [1 for i in range(batch1_num)]
    batch_method_list.extend([2 for i in range(batch2_num)])
    batch_method_list.extend([3 for i in range(batch3_num)])
    for batch_method in batch_method_list:
        if len(result_data)==cut_value:
            obj_iter_start_num = set_aug_result(db, result_data, grid, grid_id, device_id, img_path_list, obj_iter_start_num)
            result_data = []
            img_path_list = []
            f = open("iter_num.txt", 'w')
            f.write(str(obj_iter_start_num))
            f.close()
        img_path, result = augment.aug_process(grid, object_category, 1, background, DB_mask, iteration_list, aug_count)
        result_data.append(result)
        img_path_list.append(img_path)
        aug_count +=1
    obj_iter_start_num = set_aug_result(db, result_data, grid, grid_id, device_id, img_path_list, obj_iter_start_num)
    f = open("iter_num.txt", 'w')
    f.write(str(obj_iter_start_num))
    f.close()
