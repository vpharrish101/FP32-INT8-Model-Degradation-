import cv2
import numpy as np
import pandas as pd
from pathlib import Path


class ImageDegrader:

    def __init__(self, csv, src, out):
        self.df = pd.read_csv(csv)
        self.src = Path(src)
        self.out = Path(out)

    def motion_blur(self, img):
        k = np.zeros((15, 15), np.float32)
        k[7, :] = 1 / 15
        return cv2.filter2D(img, -1, k)

    def low_light(self, img):
        return ((img / 255) ** 2 * 255).astype(np.uint8)

    def jpeg30(self, img):
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def downscale50(self, img):
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), cv2.INTER_AREA)
        return cv2.resize(img, (w, h), cv2.INTER_LINEAR)

    def generate(self):
        funcs = {
            "motion_blur": self.motion_blur,
            "low_light": self.low_light,
            "jpeg30": self.jpeg30,
            "downscale50": self.downscale50,
        }

        for _, row in self.df.iterrows():
            name = row["file_name"]
            img = cv2.imread(str(self.src / name))

            for mode, func in funcs.items():
                path = self.out / mode
                path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path / name), func(img))


if __name__ == "__main__":
    ImageDegrader(
        "data/coco2017/coco_500_val.csv",
        "data/coco2017/val2017",
        "data/coco2017/degraded",
    ).generate()