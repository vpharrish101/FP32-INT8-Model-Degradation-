import cv2

from src.FP32_metrics import benchmark
from src.utils.quantizer import blur_calib
from src.utils.degrade import ImageDegrader
from src.utils.COCO_val_sampler import main as blur_csv_constructor


def main():
    blur_csv_constructor(n=1000,output="data/coco2017/coco_1000_train.csv",annotation="data/coco2017/annotations/instances_train2017.json")

    degrader=ImageDegrader(
        csv="data/coco2017/coco_1000_train.csv",
        src="data/coco2017/train2017",
        out="data/coco2017/degraded/motion_blur_calibration",)

    degrader.out.mkdir(parents=True,exist_ok=True)

    for i,(_,row) in enumerate(degrader.df.iterrows()):
        name=row["file_name"]
        img=cv2.imread(str(degrader.src/name))

        if i<300:
            out=img
            name=f"clean_{name}"
        elif i<400:
            out=degrader.motion_blur(img,5)
            name=f"blur5_{name}"
        elif i<550:
            out=degrader.motion_blur(img,9)
            name=f"blur9_{name}"
        elif i<800:
            out=degrader.motion_blur(img,15)
            name=f"blur15_{name}"
        elif i<950:
            out=degrader.motion_blur(img,21)
            name=f"blur21_{name}"
        else:
            out=degrader.motion_blur(img,31)
            name=f"blur31_{name}"

        cv2.imwrite(str(degrader.out/name),out)

    blur_calib()

    print("FP32")
    benchmark("models/yolo11n_openvino_model")
    print("-" * 80)
    print("Original INT8")
    benchmark("models/yolo11n_int8_uncalibrated_openvino_model")
    print("-" * 80)
    print("Motion-blur calibrated INT8")
    benchmark("models/yolo11n_int8_openvino_model")
    print("-" * 80)
    print("Original INT8 - motion blur")
    benchmark("models/yolo11n_int8_uncalibrated_openvino_model","motion_blur")
    print("-" * 80)
    print("Motion-calibrated INT8 - motion blur")
    benchmark("models/yolo11n_int8_openvino_model","motion_blur")


if __name__ == "__main__":
    main()