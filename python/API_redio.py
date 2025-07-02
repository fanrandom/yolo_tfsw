from fastapi import FastAPI
import cv2
import os

app = FastAPI()

def save_frames_from_rtsp(rtsp_url: str, output_folder: str, every_n_frames: int = 30):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception(f"❌ 无法连接 RTSP 流: {rtsp_url}")

    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    saved_count = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every_n_frames == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

        # 限制最多保存多少张（避免长时间运行时挂掉）
        if saved_count >= 100:
            break

    cap.release()
    return saved_count

@app.get("/extract-rtsp/")
def extract_from_rtsp(rtsp_url: str = "rtsp://127.0.0.1:8554/video", interval: int = 30):
    output_dir = "E:/azhonggonghulian/tongfangshuiwu/yolov5-master/yolov5-master/dataset"
    try:
        count = save_frames_from_rtsp(rtsp_url, output_dir, every_n_frames=interval)
        return {"message": f"✅ 成功保存 {count} 张帧图像到 {output_dir}"}
    except Exception as e:
        return {"error": str(e)}

