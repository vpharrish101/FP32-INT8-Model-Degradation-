from src.utils.COCO_val_sampler import main as sample
from src.utils.quantizer import main as quantize
from src.FP32_metrics import main as metrics


def main():
    sample()
    quantize()
    metrics()


if __name__ == "__main__":
    main()