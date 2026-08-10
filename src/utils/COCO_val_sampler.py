import pandas as pd

from pycocotools import coco

SEED=76

def main():
    coco_json=coco.COCO("data/coco2017/annotations/instances_val2017.json")
    classes=["person","bicycle","car","traffic light","stop sign"]
    cat_ids=coco_json.getCatIds(catNms=classes)
    img_ids=list(set().union(*(coco_json.getImgIds(catIds=[cat_id]) for cat_id in cat_ids)))
    images=coco_json.loadImgs(img_ids)
    df=pd.DataFrame(images)
    df.sample(n=500,random_state=SEED)[["id", "file_name"]].to_csv("data/coco2017/coco_500_val.csv",index=False)

if __name__=="__main__":
    main()