import cv2


def save_first_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
    cap.release()


# Example usage:
save_first_frame('videos\\Counter.mp4', 'first_frame.jpg')
