import cv2
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from trackers import SORTTracker

RESIZE_OUTPUT = (1530, 780)

model = RFDETRBase()
tracker = SORTTracker()
round_box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

fps_monitor = sv.FPSMonitor()


def process_frame(frame, confidence=0.3):
    detections = model.predict(frame, threshold=confidence)
    detections = detections[detections.class_id == 3]  # Filter for class_id 3 (car)
    detections = tracker.update(detections)
    fps_monitor.tick()
    labels = []
    for _, _, confidence, class_id, tracker_id, _ in detections:
        class_name = COCO_CLASSES.get(class_id, f"Class {class_id}")
        class_name = class_name.title()
        tracker_label = f"#{tracker_id}" if tracker_id != -1 else ""
        label = f"{tracker_label} {class_name} {confidence * 100:.1f}%"
        label = label.strip()
        labels.append(label)

    annotated_frame = frame.copy()
    annotated_frame = round_box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    fps = fps_monitor.fps
    annotated_frame = sv.draw_text(
        scene=annotated_frame,
        text=f"FPS: {fps:.2f}",
        text_anchor=sv.Point(x=40, y=10),
    )
    return annotated_frame


def process_stream(stream=0):
    frame_generator = sv.get_video_frames_generator(stream)
    for frame_index, frame in enumerate(frame_generator):
        annotated_frame = process_frame(frame, confidence=0.3)

        resized_frame = cv2.resize(annotated_frame, RESIZE_OUTPUT)
        cv2.imshow("ZoneVision", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break
    cv2.destroyAllWindows()


def main():
    stream = "videos/Intersection.mp4"
    process_stream(stream=stream)


if __name__ == "__main__":
    main()
