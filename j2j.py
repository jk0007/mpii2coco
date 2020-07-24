import json
import cv2
import os


# =======================================读取mpii标注=======================================
mpiipath = "./mpii/annot/train.json"  # mpii train/valid的地址
# mpiipath = "./mpii/annot/valid.json"

mpii_imgpath = "./mpii/images/"  # mpii图片的地址
writepath = "./coco/annotations/mpii2coco_train.json"  # mpii转换成coco的地址
with open(mpiipath, 'r', encoding='utf-8')as f:
    mpii = json.load(f)

# cocopath = "./coco/annotations/new_person_keypoints_train2017.json"  # 原coco train2017的地址
# with open(cocopath, 'r', encoding='utf-8')as f:
#     origin_coco = json.load(f)

# =======================================数据集转换=======================================
coco = { # 定义coco标注的总字典，包含三个子字典
    'annotations': [],
    "categories": [{"name": "person", "id": 1}],
    "images": []
}

for i in range(len(mpii)):
    print(f'process {i}*******************************************')
    # if i > 5000:
    #     break

    mpii_inf = mpii[i]  # mpii_inf是mpii.json中第i个标注
    # 必须要在for循环中定义coco.json文件中的两个字典，
    # 因为append()是浅复制，在for外面定义会导致之前append的block内容被覆盖
    annot_block = {
        "category_id": 1,
        "bbox": [],
        "id": "000000000",
        "image_id": "000000000",
        "keypoints": [],
        "iscrowd": 0,
        "num_keypoints": 0
    }
    images_block = {
        "file_name": "000000000.jpg",
        "id": 000000000
    }

    ids = ''.join(a for a in mpii_inf['image'] if a in "0123456789")
    annot_block['id'] = ids
    annot_block['image_id'] = ids
    annot_block['num_keypoints'] = sum(mpii_inf['joints_vis'])
    images_block['file_name'] = mpii_inf['image']
    images_block['id'] = ids

#           ===========================keypoints的转换============================
    for j in range(16):
        if mpii_inf['joints_vis'][j] == 1:
            annot_block['keypoints'].extend([int(mpii_inf['joints'][j][0]), int(mpii_inf['joints'][j][1]), 2])
        else:
            annot_block['keypoints'].extend([0, 0, 0])

#           =============================bbox的转换===============================
    aspect_ratio = 288/384
    center = [mpii_inf['center'][0], mpii_inf['center'][1]]
    scale = mpii_inf['scale']

    # scale = scale * 1.25
    # center[1] = center[1] + 15 * scale
    w = int(scale * 200)  # 200
    h = scale * 200  # 200
    h = int(h/aspect_ratio)
    x = int(center[0] - w * 0.5)
    y = int(center[1] - h * 0.5)
    annot_block['bbox'].extend([x, y, w, h])
#           ========================显示图片，可注释掉==============================
    img = cv2.imread(os.path.join(mpii_imgpath, images_block['file_name']))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    for k in range(16):
        cv2.circle(img,((annot_block['keypoints'][k * 3]),int(annot_block['keypoints'][k * 3 + 1])),2,(0,255,0),3)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    # img = cv2.imread(''.join(mpii_imgpath,images_block['file_name']))
    # print(''.join(mpii_imgpath,images_block['file_name']))
    print(img.shape)
    print('center,scale: {}, {}({}), {}'.format(mpii_inf['center'][0], mpii_inf['center'][1], center[1], mpii_inf['scale']))
    print('xywh: {}, {}, {}, {}'.format(x, y, w, h))
#           =============================写入文件=================================
    coco['annotations'].append(annot_block)
    coco['images'].append(images_block)
    cv2.waitKey(0)

with open(writepath, 'w', encoding='utf-8') as f:
    json.dump(coco, f, indent=4)
print('ret')
