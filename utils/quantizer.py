from ultralytics import YOLO
from pathlib import Path


def main():

    model=YOLO("models/yolo11n.pt")

    model.export(format="openvino",imgsz=640,)

    model.export(
        format="openvino",
        imgsz=640,
        quantize=8,
        data="data/coco_calib.yaml",
        fraction=0.06)

    Path("models/yolo11n_int8_openvino_model").rename(
        "models/yolo11n_int8_uncalibrated_openvino_model"
    )

def blur_calib():
    model=YOLO("models/yolo11n.pt")
    model.export(
        format="openvino",
        imgsz=640,
        quantize=8,
        data="data/blur_coco_calib.yaml",
        name="yolo11n_openvino_blurcalib",
        fraction=1.0
    )

