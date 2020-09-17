import cv2
import os
import numpy as np
from utils import *
from multiprocessing import Process

def prepare_read_model(model_dir):
    _, model_face, _, _ ,_= read_model(os.path.join(model_dir, 'aodi-Q7-SUV.obj'), read_face=True)
    model_pcs = {}
    model_tus = {}
    model_tvs = {}
    
    for obj in os.listdir(model_dir):
        if 'obj' in obj:
            # obj = 'leikesasi.obj'
            pc, _, tu, tv ,_= read_model(os.path.join(model_dir, obj))
            model_pcs[obj.split('.')[0]] = pc
            model_tus[obj.split('.')[0]] = tu
            model_tvs[obj.split('.')[0]] = tv

    return model_pcs, model_face, model_tus, model_tvs

def blend_and_fill_black_area(texture_map, iteration = 1, block_size = 5):
    part_texture = cv2.imread('Template18_new.PNG')
    res = np.copy(texture_map)
    for _ in range(iteration):
        black_area_uv = np.argwhere(((res[:, :, 0] == 0) & (texture_map[:, :, 1] == 0) & (texture_map[:, :, 2] == 0) & (part_texture[:, :, 1]!=0)))
        for i in range(black_area_uv.shape[0]):
            u = black_area_uv[i][0]
            v = black_area_uv[i][1]
            block = texture_map[u - block_size:u + block_size, v - block_size:v + block_size, :]
            res[u][v][0] = np.max(block[:, :, 0])
            res[u][v][1] = np.max(block[:, :, 1])
            res[u][v][2] = np.max(block[:, :, 2])
    
    return res

def calculate_integrity(texture_mask, recon_texture_part):
    total = np.sum(texture_mask)
    correct_part_mask = (recon_texture_part[:, :, 0]!=0) & (recon_texture_part[:, :, 1]!=0) & (recon_texture_part[:, :, 2]!=0)
    part = np.sum(correct_part_mask)
    integrity = part / total
    return integrity


def recon_part_from_apollo(image_prefix, camera_matrix, color_img, label_img, depth_img, poses, model_pcs, model_face, model_tus, model_tvs, texture_bbox_dict, texture_mask_dict):
    id_channel = label_img[:, :, 0]
    car_ids = np.unique(id_channel)
    if len(car_ids)-1 != len(poses):
        return None
    for car_id, pose in enumerate(poses):
        mask = label_img[:, :, 0] == car_id
        car_name = get_car_name(pose["name_index"])
        tu = model_tus[car_name]
        tv = model_tvs[car_name]

        pc_rgb = np.zeros((6, model_pcs[car_name].shape[1]))
        pc_rgb[0: 3] = model_pcs[car_name]

        rot_mat = get_rotation_mat(pose["a"], pose["b"], pose["c"])
        pc = np.dot(rot_mat, model_pcs[car_name])
        pc[0, :] += pose["tx"]
        pc[1, :] += pose["ty"]
        pc[2, :] += pose["tz"]
        pc_2 = np.dot(camera_matrix, pc)
        us = pc_2[0, :] / pc[2, :]
        vs = pc_2[1, :] / pc[2, :]
        zs = pc[2, :]

        texture_curr = np.zeros((2048, 2048, 3))
        # texture_curr = cv2.imread('Template18_new.PNG')
        for pid, (u, v, z) in enumerate(zip(us, vs, zs)):
            if u<0 or v<0 or u>=3384 or v>=2710:
                continue
            u = int(u)
            v = int(v)
            if mask[v, u] == False:
                continue
            if abs(depth_img[v, u] - z) < 0.1:
                pc_rgb[3, pid] = color_img[v, u, 2]
                pc_rgb[4, pid] = color_img[v, u, 1]
                pc_rgb[5, pid] = color_img[v, u, 0]
                texture_curr[tv[pid], tu[pid], 0] = color_img[v, u, 0]
                texture_curr[tv[pid], tu[pid], 1] = color_img[v, u, 1]
                texture_curr[tv[pid], tu[pid], 2] = color_img[v, u, 2]
            else:
                continue
        # save_pc_rgb('test.obj', pc_rgb)
        texture_curr = blend_and_fill_black_area(texture_curr)
        # cv2.imwrite('test_' + str(car_id) + '.png', texture_curr)
        for part_id in range(1):
            part_id = 3
            print(part_id)
            part_bbox = texture_bbox_dict[part_id]
            part_texture_curr = texture_curr[part_bbox[0]:part_bbox[2], part_bbox[1]:part_bbox[3]]
            integrity = calculate_integrity(texture_mask_dict[part_id], part_texture_curr)
            print(integrity)
            if integrity > 0.9:
                if len(os.listdir(os.path.join('part_complete', str(part_id)))) > 3000:
                    continue
                cv2.imwrite(os.path.join('part_complete', str(part_id), image_prefix + '_' + str(car_id) + '.png'), part_texture_curr)
            if 0.3 < integrity < 0.9:
                if len(os.listdir(os.path.join('part_missing', str(part_id)))) > 200:
                    continue
                missing_region_mask = (part_texture_curr[:,:,0]==0) & (part_texture_curr[:,:,1]==0) & (part_texture_curr[:,:,2]==0) & (texture_mask_dict[part_id][part_bbox[0]:part_bbox[2], part_bbox[1]:part_bbox[3]])
                part_missing_region = np.zeros(part_texture_curr.shape)
                part_missing_region[missing_region_mask] = (255,255,255)
                cv2.imwrite(os.path.join('part_missing', str(part_id), image_prefix + '_' + str(car_id) + '_' + str(integrity) + '.png'), part_missing_region)
                cv2.imwrite(os.path.join('part_test_data', str(part_id), image_prefix + '_' + str(car_id) + '_' + str(integrity) + '.png'), part_texture_curr)
                
def get_part_recon(img_list, camera_matrix, model_pcs, model_face, model_tus, model_tvs, texture_bbox_dict, texture_mask_dic):
    image_dir = '/media/miao/新加卷/view_changed/crossroads/apollo'
    pose_dir = '/media/miao/新加卷/view_changed/resource/Labelled_Pose/5'
    label_dir = '/media/miao/新加卷/view_changed/resource/Render_ID/i5'
    depth_dir = '/media/miao/新加卷/view_changed/resource/Render_Depth/d5'
    model_dir = '/media/miao/新加卷/view_changed/crossroads/reconstruction'
    texture_map_path = 'Template18_new.PNG'
    img_list.reverse()
    for image_file in img_list:
        # image_file = '171206_034559609_Camera_5.jpg'
        print(image_file)
        image_prefix = image_file.split('.')[0]
        if not os.path.exists(os.path.join(label_dir, image_prefix + '-label.png')):
            continue
        color_img = cv2.imread(os.path.join(image_dir, image_prefix + '.jpg'))
        label_img = cv2.imread(os.path.join(label_dir, image_prefix + '-label.png'))
        depth_img = cv2.imread(os.path.join(depth_dir, image_prefix + '-depth.png'), cv2.IMREAD_UNCHANGED) / 100
        poses = read_pose_file(os.path.join(pose_dir, image_prefix + '.txt'))
        car_pcs = recon_part_from_apollo(image_prefix, camera_matrix, color_img, label_img, depth_img, poses, model_pcs, model_face, model_tus, model_tvs, texture_bbox_dict, texture_mask_dict)
        if car_pcs is None:
            continue

 
if __name__ == "__main__":
    image_dir = '/media/miao/新加卷/view_changed/crossroads/apollo'
    pose_dir = '/media/miao/新加卷/view_changed/resource/Labelled_Pose/5'
    label_dir = '/media/miao/新加卷/view_changed/resource/Render_ID/i5'
    depth_dir = '/media/miao/新加卷/view_changed/resource/Render_Depth/d5'
    model_dir = '/media/miao/新加卷/view_changed/crossroads/reconstruction'
    texture_map_path = 'Template18_new.PNG'
    model_pcs, model_face, model_tus, model_tvs = prepare_read_model(model_dir)
    fx = 2304.54786556982
    fy = 2305.875668062
    cx = 1686.23787612802
    cy = 1354.98486439791
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
    texture_mask_dict = get_part_mask(texture_map_path)
    texture_bbox_dict = get_part_patch_box(texture_map_path)

    image_list = os.listdir(image_dir)

    num_of_worker = 10
    print(len(image_list))
    num_per_worker = len(image_list) // num_of_worker
    processes = []
    for i in range(num_of_worker):
        if i == num_of_worker - 1:
            p = Process(target=get_part_recon, args=(image_list[i * num_per_worker:], camera_matrix, model_pcs, model_face, model_tus, model_tvs, texture_bbox_dict, texture_mask_dict))
        else:
            p = Process(target=get_part_recon, args=(image_list[i * num_per_worker:(i + 1) * num_per_worker], camera_matrix, model_pcs, model_face, model_tus, model_tvs, texture_bbox_dict, texture_mask_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
   