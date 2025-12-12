import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    # 原版使用 ResNet50+FPN
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1  # 不包含背景
    box_thresh = 0.3 # 建议设置在 0.3 左右，观察召回能力
    
    # --- 路径配置 ---
    weights_path = "./save_weights/model_25.pth"  # 请确保这是 ResNet50 的权重
    img_dir = "./test_image"                      # 测试图片文件夹
    output_dir = "./test_result_resnet50"         # 结果保存路径
    label_json_path = './coco91_indices.json'
    # ---------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载模型
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)
    assert os.path.exists(weights_path), f"file {weights_path} does not exist."
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # 加载类别标签
    assert os.path.exists(label_json_path), f"json file {label_json_path} does not exist."
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # 获取图片列表
    supported = (".jpg", ".jpeg", ".png", ".bmp")
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(supported)]
    
    print(f"开始预测 {len(img_list)} 张图片...")
    model.eval()
    
    with torch.no_grad():
        for i, img_name in enumerate(img_list):
            img_path = os.path.join(img_dir, img_name)
            original_img = Image.open(img_path).convert('RGB')

            # 预处理
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            # 第一次运行热身
            if i == 0:
                model(img.to(device))

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print(f"[{i+1}/{len(img_list)}] {img_name} cost: {t_end - t_start:.3f}s")

            # 解析结果
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)

            if len(predict_boxes) == 0:
                print(f"  -> {img_name}: 无目标")
                continue

            # 绘制结果
            plot_img = draw_objs(original_img,
                                 boxes=predict_boxes,
                                 classes=predict_classes,
                                 scores=predict_scores,
                                 masks=predict_mask,
                                 category_index=category_index,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)
            
            # 保存
            plot_img.save(os.path.join(output_dir, img_name))

    print(f"预测完成！结果已保存在: {output_dir}")

if __name__ == '__main__':
    main()