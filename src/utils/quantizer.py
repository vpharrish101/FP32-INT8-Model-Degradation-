from ultralytics import YOLO

def main():

    model=YOLO("models/yolo11n.pt")

    model.export(
        format="openvino",
        imgsz=640,)

    model.export(
        format="openvino",
        imgsz=640,
        quantize=8,
        data="data/coco_calib.yaml",
        fraction=0.06)

if __name__=="__main__":
    main()