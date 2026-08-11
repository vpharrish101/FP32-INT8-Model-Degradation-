import pandas as pd

from pycocotools import coco

SEED=76



def main(
    n=500,
    output=None,
    annotation="data/coco2017/annotations/instances_val2017.json",):
    coco_json=coco.COCO(annotation)

    classes=[
        "person",
        "bicycle",
        "car",
        "traffic light",
        "stop sign",
    ]
    
    cat_ids=coco_json.getCatIds(catNms=classes)
    img_ids=list(set().union(*(coco_json.getImgIds(catIds=[cat_id]) for cat_id in cat_ids)))
    images=coco_json.loadImgs(img_ids)
    df=pd.DataFrame(images)
    df.sample(n=n,random_state=SEED)[["id", "file_name"]].to_csv(output, index=False)