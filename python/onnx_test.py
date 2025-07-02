import onnxruntime
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch

# 1. 定义模型路径和类别标签
onnx_file_name = "E:/azhonggonghulian/tongfangshuiwu/yolov5-master/yolov5-master/task33_best.onnx"
class_names = ["abnormal", "normal"]  # 替换为你的实际类别名称

# 2. 图片预处理函数（使用 OpenCV）
def preprocess_image_cv2(image_path, input_size=(1312, 1312)):
    """使用 OpenCV 加载图像并转换为模型需要的格式"""
    image = cv2.imread(image_path)  # BGR
    image1 = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    image = cv2.resize(image, input_size)

    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 HWC 转为 CHW，并除以 255
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度
    return input_tensor.numpy(), image1

# 3. 加载 ONNX 模型
ort_session = onnxruntime.InferenceSession(onnx_file_name)

# 4. 指定图片路径
input_image_path = "E:/azhonggonghulian/tongfangshuiwu/dataset/tongfangdataset/task33/selftest/task33_173.jpg"

# 5. 预处理图片并推理
input_img, input_img1 = preprocess_image_cv2(input_image_path)
ort_inputs = {ort_session.get_inputs()[0].name: input_img}
ort_output = ort_session.run(None, ort_inputs)[0]

# 6. 解析结果
predicted_class_idx = np.argmax(ort_output, axis=1)[0]
predicted_class_name = class_names[predicted_class_idx]
confidence = float(np.max(ort_output))


# 7. 打印结果
print(f"Input Image: {input_image_path}")
print(f"Predicted Class: {predicted_class_name} (Index: {predicted_class_idx})")
print(f"Confidence: {confidence:.4f}")

# 9. 在图片上绘制预测结果并保存
output_img = input_img1.copy()
text = f"{predicted_class_name} ({confidence:.2f})"
cv2.putText(output_img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
# output_img_path = "/root/fwm/yolov5-master/output.jpg"
# cv2.imwrite(output_img_path, output_img)

# 8. 可视化显示（OpenCV）
#cv2.imshow("Input Image", cv2.imread(input_image_path))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite("/root/fwm/yolov5-master/output.jpg", input_img)
