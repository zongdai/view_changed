import numpy as np
import cv2
import math

def get_texture_part_color():
    color_table = [
		[247, 77, 149],
		[32, 148, 9],
		[166 ,104, 6],
		[7 ,212, 133],
		[1, 251, 1],
		[2, 2, 188],
		[219, 251, 1],
		[96, 94, 92],
		[229, 114, 84],
		[216, 166, 255],
		[113, 165, 0],
		[8, 78, 183],
		[112, 252, 57],
        [5, 28, 126],
		[100, 111, 156],
		[140, 60, 39],
        [75, 13, 159],
        [188, 110, 83]
	]

    return color_table

def get_part_mask(texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    parts_color = get_texture_part_color()
    texture_mask_dict = {}
    for i, color in enumerate(parts_color):
        texture_mask_dict[i] = texture_map[:, :, 2] == color[0]
    return texture_mask_dict


def get_part_patch_box(texture_map_path):
    texture_map = cv2.imread(texture_map_path)
    parts_color = get_texture_part_color()
    bboxes = {}
    for id, color in enumerate(parts_color):
        area_uv = np.argwhere(texture_map[:, :, 2] == color[0])
        # print(area_uv.shape)
        min_v = np.min(area_uv[:, 0])
        min_u = np.min(area_uv[:, 1])
        max_v = np.max(area_uv[:, 0])
        max_u = np.max(area_uv[:, 1])
        bboxes[id] = [min_v, min_u, max_v, max_u]
        # cv2.imwrite('part_vis/' + str(id) + '.png', texture_map[min_v:max_v + 1, min_u:max_u + 1])
    # print(bboxes)
    return bboxes

def read_pose_file(pose_file):

    with open(pose_file) as f:
        car_infos = []
        line = f.readline()
        while(line):
            info = {}
            items = line.split('  ')
            # print(items)
            info["name_index"] = int(items[0])
            info["a"] = float(items[1])
            info["b"] = float(items[2])
            info["c"] = float(items[3])
            info["tx"] = float(items[4])
            info["ty"] = float(items[5])
            info["tz"] = float(items[6].split('\n')[0])
            # print(info)
            car_infos.append(info)
            line = f.readline()
    return car_infos

def read_model(point_cloud_path, read_face=False, read_color=False, read_vt=False, texture_size=[2048, 2048], scale=1.0):
    x = []
    y = []
    z = []
    u = []
    v = []
    rgb = []
    face_index = []
    with open(point_cloud_path) as f:
        line = f.readline()

        while line:
            if line[0] == 'v' and line[1] == ' ':
                items = line.split()
                x.append(float(items[1]) * scale)
                y.append(float(items[2]) * scale)
                z.append(float(items[3].replace("\n", "")) * scale)
                if read_color:
                    r = float(items[4])
                    g = float(items[5])
                    b = float(items[6].replace("\n", ""))
                    rgb.append([r, g, b])

            elif line[0] == 'f' and read_face:
                item = line.split(' ')
                f1 = int(item[1].split('/')[0]) - 1
                f2 = int(item[2].split('/')[0]) - 1
                f3 = int(item[3].split('/')[0]) - 1
                if len(item) == 5:
                    f4 = int(item[4].split('/')[0]) - 1
                    face_index.append([f1, f2, f3, f4])
                else:
                    face_index.append([f1, f2, f3])
            elif line[0] == 'v' and line[1] == 't':
                items = line.split(' ')
                u.append(int((float(items[1])) * texture_size[0]))
                v.append(int((1 - float(items[2])) * texture_size[1]))
            line = f.readline()

    return np.array([x, y, z]), np.array(face_index), np.array(u), np.array(v), np.array(rgb).T

def get_car_name(index):
    model_name = {}
    model_name[0] = "baojun-310-2017"
    model_name[1] = "biaozhi-3008"
    model_name[2] = "biaozhi-liangxiang"
    model_name[3] = "bieke-yinglang-XT"
    model_name[4] = "biyadi-2x-F0"
    model_name[5] = "changanbenben"
    model_name[6] = "dongfeng-DS5"
    model_name[7] = "feiyate"
    model_name[8] = "fengtian-liangxiang"
    model_name[9] = "fengtian-MPV"
    model_name[10] = "jilixiongmao-2015"
    model_name[11] = "lingmu-aotuo-2009"
    model_name[12] = "feiyate"
    model_name[13] = "lingmu-SX4-2012"
    model_name[14] = "sikeda-jingrui"
    model_name[15] = "fengtian-weichi-2006"
    model_name[16] = "037-CAR02"
    model_name[17] = "aodi-a6"
    model_name[18] = "baoma-330"
    model_name[19] = "baoma-530"
    model_name[20] = "baoshijie-paoche"
    model_name[21] = "bentian-fengfan"
    model_name[22] = "biaozhi-408"
    model_name[23] = "biaozhi-508"
    model_name[24] = "bieke-kaiyue"
    model_name[25] = "fute"
    model_name[26] = "haima-3"
    model_name[27] = "kaidilake-CTS"
    model_name[28] = "leikesasi"
    model_name[29] = "mazida-6-2015"
    model_name[30] = "MG-GT-2015"
    model_name[31] = "oubao"
    model_name[32] = "qiya"
    model_name[33] = "rongwei-750"
    model_name[34] = "supai-2016"
    model_name[35] = "xiandai-suonata"
    model_name[36] = "yiqi-benteng-b50"
    model_name[37] = "bieke"
    model_name[38] = "biyadi-F3"
    model_name[39] = "biyadi-qin"
    model_name[40] = "dazhong"
    model_name[41] = "dazhongmaiteng"
    model_name[42] = "dihao-EV"
    model_name[43] = "dongfeng-xuetielong-C6"
    model_name[44] = "dongnan-V3-lingyue-2011"
    model_name[45] = "dongfeng-yulong-naruijie"
    model_name[46] = "019-SUV"
    model_name[47] = "036-CAR01"
    model_name[48] = "aodi-Q7-SUV"
    model_name[49] = "baojun-510"
    model_name[50] = "baoma-X5"
    model_name[51] = "baoshijie-kayan"
    model_name[52] = "beiqi-huansu-H3"
    model_name[53] = "benchi-GLK-300"
    model_name[54] = "benchi-ML500"
    model_name[55] = "fengtian-puladuo-06"
    model_name[56] = "fengtian-SUV-gai"
    model_name[57] = "guangqi-chuanqi-GS4-2015"
    model_name[58] = "jianghuai-ruifeng-S3"
    model_name[59] = "jili-boyue"
    model_name[60] = "019-SUV"
    model_name[61] = "linken-SUV"
    model_name[62] = "lufeng-X8"
    model_name[63] = "qirui-ruihu"
    model_name[64] = "rongwei-RX5"
    model_name[65] = "sanling-oulande"
    model_name[66] = "sikeda-SUV"
    model_name[67] = "Skoda_Fabia-2011"
    model_name[68] = "xiandai-i25-2016"
    model_name[69] = "yingfeinidi-qx80"
    model_name[70] = "yingfeinidi-SUV"
    model_name[71] = "benchi-SUR"
    model_name[72] = "biyadi-tang"
    model_name[73] = "changan-CS35-2012"
    model_name[74] = "changan-cs5"
    model_name[75] = "changcheng-H6-2016"
    model_name[76] = "dazhong-SUV"
    model_name[77] = "dongfeng-fengguang-S560"
    model_name[78] = "dongfeng-fengxing-SX6"
    return model_name[index]

def get_rotation_mat(a, b, c):
    rotation = np.zeros((3,3))
    rotation[0][0] = math.cos(c) * math.cos(b)
    rotation[0][1] = -math.sin(c) * math.cos(a) + math.cos(c) * math.sin(b) * math.sin(a)
    rotation[0][2] = math.sin(a) * math.sin(c) + math.cos(c) * math.sin(b) * math.cos(a)
    rotation[1][0] = math.cos(b) * math.sin(c)
    rotation[1][1] = math.cos(c) * math.cos(a) + math.sin(c) * math.sin(b) * math.sin(a)
    rotation[1][2] = -math.sin(a) * math.cos(c) + math.cos(a) * math.sin(b) * math.sin(c)
    rotation[2][0] = -math.sin(b)
    rotation[2][1] = math.cos(b) * math.sin(a)
    rotation[2][2] = math.cos(a) * math.cos(b)
    return rotation

def save_pc_rgb(pc_path, pc_rgb):
    with open(pc_path, 'w') as f:
        for i in range(pc_rgb.shape[1]):
            f.write('v ' + str(pc_rgb[0, i]) + ' ' + str(pc_rgb[1, i]) + ' '+ str(pc_rgb[2, i]) + ' '+ str(pc_rgb[3, i]) + ' '+ str(pc_rgb[4, i]) + ' '+ str(pc_rgb[5, i]) + '\n')
        