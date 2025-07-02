from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import base64
import onnxruntime
import numpy as np
import cv2
import torchvision.transforms as transforms

app = FastAPI()

# 定义输入数据结构：一张图片 + 时间戳
# 定义单个Item的模型
class Item(BaseModel):
    mediaKey: str  # 字符串时间戳（如 "2023-01-01T12:00:00"）
    # image_base64: str       # Base64 编码的图片
    timestamp: str  # 标签（如 "cat"）
    streamUrl: str
    detectType: str

# 定义整个请求体的模型
class BatchRequest(BaseModel):
    items: List[Item]    # 注意字段名是items，不是images


# 加载模型 & 类别
onnx_file_name = "task33_best.onnx"
class_names = ["abnormal", "normal"]
ort_session = onnxruntime.InferenceSession(onnx_file_name)

# 图像预处理函数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
def batch_predict(request: BatchRequest):
    abnormal_results = []


    for item in request.items:
        try:
            # 解码 base64 图像
            img_bytes = base64.b64decode(item.image_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 直接解码为RGB格式

            # 尺寸调整（1312x1312，使用双三次插值）
            img = cv2.resize(img, (1312, 1312), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img).unsqueeze(0).numpy()

            # 模型推理
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
            ort_output = ort_session.run(None, ort_inputs)[0]
            predicted_class_idx = np.argmax(ort_output, axis=1)[0]
            confidence = float(np.max(ort_output))
            predicted_class_name = class_names[predicted_class_idx]

            if predicted_class_idx == 0:  # 异常
                # 在图像上加文本
                img_np = np.array(img)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.putText(img_np, f"{predicted_class_name} ({confidence:.2f})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                _, buffer = cv2.imencode(".jpg", img_np)
                img_b64_result = base64.b64encode(buffer).decode("utf-8")

                abnormal_results.append({
                    "timestamp": item.timestamp,
                    'isException': 'True',
                    # "confidence": round(confidence, 4),
                    "exceptionDetectType": predicted_class_name,
                    "exceptionBase64Image": img_b64_result
                })
                break

        except Exception as e:
            abnormal_results.append({
                "timestamp": item.timestamp,
                "error": str(e)
            })

    if not abnormal_results:
        return {"message": "✅ 所有图像均为正常"}
    else:
        return {
            "message": f"⚠️ 共检测到 {len(abnormal_results)} 张异常图像",
            "results": abnormal_results
        }
