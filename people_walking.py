import cv2
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from trackers import SORTTracker

RESIZE_OUTPUT = (1530, 780)

model = RFDETRBase()
tracker = SORTTracker()
round_box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=0.4, text_padding=4)

fps_monitor = sv.FPSMonitor()


def process_frame(frame, confidence=0.3):
    detections = model.predict(frame, threshold=confidence)
    detections = detections[detections.class_id == 1]  # Filter for class_id 1 (person)
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


def main():
    source = "videos/walks.mp4"
    target = "output/walks_annotated.mp4"

    def callback(frame, frame_index):
        annotated_frame = process_frame(frame, confidence=0.25)

        resized_frame = cv2.resize(annotated_frame, RESIZE_OUTPUT)
        cv2.imshow("ZoneVision", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
        return annotated_frame

    sv.process_video(
        source_path=source,
        target_path=target,
        callback=callback,
        show_progress=True,
    )

    cv2.destroyAllWindows()

    print(f"Saved annotated video to {target}")


if __name__ == "__main__":
    main()
