from src.utils.COCO_val_sampler import main as sample
from src.utils.degrade import main as degrade
from src.utils.quantizer import main as quantize
from src.FP32_metrics import main as metrics
from src.Degraded_metrics import main as degraded_metrics
from src.blur_calibration import main as intervention

def main():
    sample(n=500,output="data/coco2017/coco_500_val.csv",annotation="data/coco2017/annotations/instances_val2017.json",)
    quantize()
    metrics()
    degrade()
    degraded_metrics()
    intervention()

if __name__ == "__main__":
    main()