import cv2

def detect_faces(image_path):
    # 加载预训练的 Haar 特征分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 读取图像
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示图像
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存处理后的图片
    output_path = "images/output.jpg"  # 你可以修改路径和文件名
    cv2.imwrite(output_path, image)
    print(f"处理后的图片已保存到 {output_path}")

if __name__ == "__main__":
    image_path = "images/sample.jpg"  # 确保图片路径正确
    detect_faces(image_path)
