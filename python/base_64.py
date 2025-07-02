import requests
import base64
import json

# 读取图片并编码为 base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        print(1)
        return base64.b64encode(img_file.read()).decode("utf-8")

# 构造多个样本
data = {

        "items":[
            {

            "image_base64": encode_image_to_base64("E:/azhonggonghulian/tongfangshuiwu/dataset/tongfangdataset/task33/selftest/task33_200.jpg"),
            "timestamp": "2024-06-01T10:00:00Z", # 时间戳


        },
        {

            "image_base64": encode_image_to_base64("E:/azhonggonghulian/tongfangshuiwu/dataset/tongfangdataset/task33/selftest/test4.png"),
            "timestamp": "2024-06-01T10:00:00Z", # 时间戳


        }
    ]
}
data1 = {

        "items":[
            {
            "mediaKey":"task33", # 视频流标识
            "image_base64": encode_image_to_base64("E:/azhonggonghulian/tongfangshuiwu/dataset/tongfangdataset/task33/selftest/task33_200.jpg"),
            "timestamp": "2024-06-01T10:00:00Z", # 时间戳
            "streamUrl":"rtsp://", # 原始流地址
            "detectType":"1,2" # 检测类型
        },
        {
            "mediaKey":"task33", # 视频流标识
            "image_base64": encode_image_to_base64("E:/azhonggonghulian/tongfangshuiwu/dataset/tongfangdataset/task33/selftest/test4.png"),
            "timestamp": "2024-06-01T10:00:00Z", # 时间戳
            "streamUrl":"rtsp://", # 原始流地址
            "detectType":"1,2" # 检测类型
        }
    ]
}

# 写入json
with open("E:/azhonggonghulian/tongfangshuiwu/yolov5-master/yolov5-master/tfsw_test.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)  # indent 参数用于美化格式

print("字典已保存到 output.json")
'''print(2)
response = requests.post("http://10.2.12.90:8000/predict", json=data1)
print(3)
print(response.status_code)
print("Response:", json.dumps(response.json(), indent=2, ensure_ascii=False))'''

'''from pydantic import BaseModel
from typing import List
import base64
import onnxruntime
import numpy as np
import cv2
import torchvision.transforms as transforms
import time
class Item(BaseModel):
    mediaKey: str  # 字符串时间戳（如 "2023-01-01T12:00:00"）
    image_base64: str       # Base64 编码的图片
    timestamp: str  # 标签（如 "cat"）
    streamUrl: str
    detectType: str

# 定义整个请求体的模型
class BatchRequest(BaseModel):
    items: List[Item]    # 注意字段名是items，不是images


# 加载模型 & 类别
onnx_file_name_list = {'task33':"task33_best.onnx", 'task34':"task34_best.onnx"}
#onnx_file_name = "task33_best.onnx"
class_names = ["abnormal", "normal"]
# ort_session = onnxruntime.InferenceSession(onnx_file_name)

# 图像预处理函数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

abnormal_results = []
request = BatchRequest(items = data1)

for item in request.items:
    try:
        start_time = time.time()
        # 解码 base64 图像
        img_bytes = base64.b64decode(item.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 直接解码为RGB格式

        # 尺寸调整（1312x1312，使用双三次插值）
        img = cv2.resize(img, (1312, 1312), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img).unsqueeze(0).numpy()

        # 选择模型，模型推理
        onnx_file_name = onnx_file_name_list[f'{item.mediaKey}']
        ort_session = onnxruntime.InferenceSession(onnx_file_name)
        
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        ort_output = ort_session.run(None, ort_inputs)[0]
        predicted_class_idx = np.argmax(ort_output, axis=1)[0]
        confidence = float(np.max(ort_output))
        predicted_class_name = class_names[predicted_class_idx]

        end_time = time.time()  # 记录结束时间

        elapsed_time = end_time - start_time
        print(f"函数运行时间: {elapsed_time:.4f} 秒")  # 保留4位小数

        if predicted_class_idx == 0:  # 异常
            # 在图像上加文本
            img_np = np.array(img)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.putText(img_np, f"{predicted_class_name} ({confidence:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            _, buffer = cv2.imencode(".jpg", img_np)
            img_b64_result = base64.b64encode(buffer).decode("utf-8")

            abnormal_results.append({
                #"mediaKey":item.mediaKey, # 视频流类型
                "timestamp": item.timestamp, # 时间戳
                'isException': 'True', # 异常为True
                # "confidence": round(confidence, 4),
                "exceptionBase64Image": img_b64_result, # 图片base64
                "exceptionDetectType": predicted_class_name # 预测类别
                })
            break
    except Exception as e:
        abnormal_results.append({
            "timestamp": item.timestamp,
            "error": str(e)
        })'''

