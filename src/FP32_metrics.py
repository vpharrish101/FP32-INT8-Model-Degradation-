import time
import ultralytics

import numpy as np
import pandas as pd
import pycocotools.coco
import pycocotools.cocoeval

import openvino as ov

ov.Core().set_property(
    "CPU",
    {"INFERENCE_NUM_THREADS": 1}
)

def benchmark(MODEL_PATH,DEGRADE_ID=None):
    y_vino=ultralytics.YOLO(MODEL_PATH,task="detect")
    y_csv=pd.read_csv("data/coco2017/coco_500_val.csv")
    y_coco=pycocotools.coco.COCO("data/coco2017/annotations/instances_val2017.json")


    TARGET_CLASSES={
        0:1,    #person
        1:2,    #bicycle
        2:3,    #car
        9:10,   #traffic light
        11:13,  #stop sign
        }


    predictions=[]
    latencies=[]
    for _, row in y_csv.iterrows():

        image_id=int(row["id"])
        if DEGRADE_ID:
            img_path = f"data/coco2017/degraded/{DEGRADE_ID}/{row['file_name']}"
        else:
            img_path = f"data/coco2017/val2017/{row['file_name']}"

        t0=time.perf_counter()
        result=y_vino(
            img_path,
            device="cpu",
            verbose=False)[0]

        latencies.append((time.perf_counter()-t0)*1000)
        boxes=result.boxes

        for box, score, cls in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy()):

            cls=int(cls)

            if cls not in TARGET_CLASSES:
                continue

            x1,y1,x2,y2=box

            predictions.append({
                "image_id": image_id,
                "category_id": TARGET_CLASSES[cls],
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                    ],
                "score": float(score)
            })

    latencies=np.array(latencies[10:]) 

    print(f"Mean latency:   {latencies.mean():.2f} ms")
    print(f"Median latency: {np.median(latencies):.2f} ms")
    print(f"P95 latency:    {np.percentile(latencies, 95):.2f} ms")
    print(f"FPS:            {1000 / latencies.mean():.2f}")


    results=y_coco.loadRes(predictions)
    evaluator=pycocotools.cocoeval.COCOeval(
        y_coco,
        results,
        "bbox")

    evaluator.params.imgIds=(y_csv["id"].astype(int).tolist())
    evaluator.params.catIds=list(TARGET_CLASSES.values())


    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    names={
        1: "person",
        2: "bicycle",
        3: "car",
        10: "traffic light",
        13: "stop sign",}

    precision=evaluator.eval["precision"]

    for i, cat_id in enumerate(evaluator.params.catIds):
        p=precision[:,:,i,0,-1]
        p=p[p>-1]
        ap=np.mean(p)
        print(f"{names[cat_id]:15s} AP: {ap:.3f}")

def main():
    print("FP32")
    benchmark("models/yolo11n_openvino_model")
    print("------------------------------------------------------------------------------")
    print("INT8")
    benchmark("models/yolo11n_int8_openvino_model")
    


if __name__=="__main__":
    main()