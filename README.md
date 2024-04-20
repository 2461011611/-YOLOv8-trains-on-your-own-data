# -YOLOv8-trains-on-your-own-data
Ultralytics YOLOv8 trains on your own data，Simple tutorial

可以参考原地址：https://github.com/ultralytics/ultralytics

# 需要更改：

 标注数据：     /root/autodl-tmp/ultralytics-main/ultralytics-main/ultralytics/cfg/datasets/data
 yaml文件   /root/autodl-tmp/ultralytics-main/ultralytics-main/ultralytics/cfg/datasets/coco.yaml

# 训练：

下载权重;https://github.com/ultralytics/ultralytics?tab=readme-ov-file  

![image](https://github.com/2461011611/-YOLOv8-trains-on-your-own-data/assets/118686100/7615e81f-dd8c-44a4-b0d6-ec74207b7dd4)

直接运行就可以，详细代码可以参考train.py

出现这个就说明训练成功了

![image](https://github.com/2461011611/-YOLOv8-trains-on-your-own-data/assets/118686100/1a3bd01c-111b-43bb-b096-a9db48cabd63)


# 使用训练好的权重进行推理、预测：


修改对应的路径就可以


![image](https://github.com/2461011611/-YOLOv8-trains-on-your-own-data/assets/118686100/5a88c97a-9a8f-45f7-8384-acb53aa77e73)


详细代码可以参考predict.py
