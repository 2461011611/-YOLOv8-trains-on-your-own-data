import argparse
import os
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description='YOLO inference')
parser.add_argument('--source', type=str, default='/root/autodl-tmp/datasets/datasets/books_images_all-crop', help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='/root/autodl-tmp/datasets/datasets/outputall', help='output folder')  # output folder
parser.add_argument('--batch_size', type=int, default=2, help='batch size for inference')  # batch size
args = parser.parse_args()

# 加载模型
model = YOLO('/root/autodl-tmp/ultralytics-main/ultralytics-main/ultralytics/runs/detect/train/weights/best.pt')

# Run inference on the source
image_folder = args.source
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 保存结果到输出文件夹
output_folder = args.output
os.makedirs(output_folder, exist_ok=True)
print("Output folder:", output_folder)

# 使用 tqdm 显示处理进度
total_images = len(image_files)
with tqdm(total=total_images, desc="Processing images") as pbar:
    for i in range(0, total_images, args.batch_size):
        batch_files = image_files[i:i + args.batch_size]
        batch_results = model(batch_files, stream=True, conf=0.8)
        
        for result, file_path in zip(batch_results, batch_files):  # 在此处添加文件路径迭代
            for j, bbox in enumerate(result.boxes.data):
                # 检查检测框是否为空
                if bbox is None:
                    print(f"No detections in image {file_path}")
                    continue
                
                # 提取边界框的坐标和类别
                x1, y1, x2, y2, conf, cls = bbox.int().tolist()  # 将Tensor类型的坐标转换为整数类型列表
                
                # 裁剪原始图像
                original_image = Image.open(file_path)  # 使用文件路径而不是索引
                cropped_image = original_image.crop((x1, y1, x2, y2))
                
                # 创建子文件夹
                class_folder = os.path.join(output_folder, f'class_{cls}')
                os.makedirs(class_folder, exist_ok=True)
                
                # 保存裁剪后的图像到子文件夹中
                filename = os.path.join(class_folder, f'result_{i}_{j}.png')
                cropped_image.save(filename)
            
        # 更新进度条
        pbar.update(len(batch_files))
