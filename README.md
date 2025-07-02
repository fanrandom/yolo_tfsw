# yolo_tfsw
# YOLO多场景异常检测API服务

基于FastAPI和YOLO ONNX的多场景异常检测API服务，支持三个场景的异常分类。

## 🚀 功能特性

- **多场景支持**: 支持三个不同场景的YOLO ONNX模型（scene1, scene2, scene3）
- **ONNX优化**: 使用ONNX Runtime实现高性能跨平台推理
- **灵活输入**: 支持文件上传和Base64编码图片输入
- **RESTful API**: 标准REST API接口
- **自动文档**: 内置Swagger UI自动生成的API文档
- **Docker部署**: 一键部署，ONNX模型内置无需外部依赖

## 📁 项目结构

```
.
├── app.py                # FastAPI应用主文件
├── requirements.txt      # Python依赖
├── Dockerfile           # Docker镜像构建文件
├── docker-compose.yml   # Docker Compose配置
├── test_api.py          # API测试脚本
├── README.md            # 项目说明文档
└── models/              # YOLO ONNX模型目录 (内置到镜像)
    ├── scene1_model.onnx
    ├── scene2_model.onnx
    └── scene3_model.onnx
```

## 🛠️ 快速部署

### 前提条件
- Docker >= 20.10
- Docker Compose >= 1.29

### 一键启动
```bash
# 构建并启动服务
docker-compose up -d

# 验证服务
curl http://localhost:5000/
```

## 📚 API接口

### 基础URL
`http://localhost:5000`

### 主要接口

#### 1. 健康检查
```http
GET /
```

#### 2. 获取模型信息
```http
GET /api/models
```

#### 3. 文件上传预测
```http
POST /api/predict/{scene_name}
```
- `scene_name`: scene1, scene2, scene3

#### 4. Base64预测
```http
POST /api/predict/{scene_name}/base64
```

### API文档
- **Swagger UI**: http://localhost:5000/docs

## 📝 使用示例

### curl示例
```bash
# 文件上传
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict/scene1

# Base64上传
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -i image.jpg)'"}' \
  http://localhost:5000/api/predict/scene1/base64
```

### Python示例
```python
import requests

# 文件上传方式
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict/scene1', files=files)
print(response.json())
```

## 🧪 测试
```bash
python test_api.py
```

## 🔧 管理命令
```bash
# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart
``` 
