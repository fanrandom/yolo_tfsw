import cv2
import os

def images_to_video(image_folder, output_path, fps=25):
    # 获取并排序所有图片（支持 jpg 和 png）
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    if not images:
        print("❌ 没有图片！")
        return

    # 使用第一张图确定尺寸
    first_img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = first_img.shape[:2]
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ 跳过无法读取的图像: {img_path}")
            continue

        # 自动调整尺寸以匹配第一张图
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        video_writer.write(img)

# 示例用法
if __name__ == "__main__":
    image_folder = "E:/azhonggonghulian/tongfangshuiwu/dataset/tongfangdataset/task33/selftest"  # 替换为你的图片文件夹路径
    output_video = "E:/azhonggonghulian/tongfangshuiwu/dataset/tongfangdataset/task33/task33_test.mp4"        # 输出视频路径
    images_to_video(image_folder, output_video, fps=25)
