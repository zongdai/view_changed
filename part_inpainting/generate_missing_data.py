import os
import cv2
import random
import numpy as np

part_id_list = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
part_id_list = [3]
for part_id in part_id_list:
    print(part_id)
    img_complete_dir = os.path.join('part_complete', str(part_id))
    img_missing_dir = os.path.join('part_missing', str(part_id))

    img_complete_list = os.listdir(img_complete_dir)
    img_missing_list = os.listdir(img_missing_dir)

    for img in img_complete_list:
        print(img)
        img_raw = cv2.imread(os.path.join(img_complete_dir, img))
        img_missing = cv2.imread(os.path.join(img_missing_dir, img_missing_list[random.randint(0, len(img_missing_list) - 1)]))
        missing_mask = img_missing[:, :, 0] == 255
        img_ge = np.copy(img_raw)
        img_ge[:,:,0][missing_mask] = 255
        img_ge[:,:,1][missing_mask] = 255
        img_ge[:,:,2][missing_mask] = 255
        cv2.imwrite(os.path.join('part_train_data', str(part_id), img), img_ge)