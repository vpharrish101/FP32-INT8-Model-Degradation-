from src.FP32_metrics import benchmark

def main():
    print(
        """
        Test-1: Motion blur (15x15 horizontal kernel)
        .
        .
        .
        .
        .
        """
    )
    print("FP32")
    benchmark("models/yolo11n_openvino_model",DEGRADE_ID="motion_blur")
    print("------------------------------------------------------------------------------")
    print("INT8")
    benchmark("models/yolo11n_int8_openvino_model",DEGRADE_ID="motion_blur")

    print(
        """
        Test-2: Low Light (Gamma correction with γ=2.0)
        .
        .
        .
        .
        .
        """
    )
    print("FP32")
    benchmark("models/yolo11n_openvino_model",DEGRADE_ID="low_light")
    print("------------------------------------------------------------------------------")
    print("INT8")
    benchmark("models/yolo11n_int8_openvino_model",DEGRADE_ID="low_light")

    print(
        """
        Test-3: JPEG Compression (Lossy)
        .
        .
        .
        .
        .
        """
    )
    print("FP32")
    benchmark("models/yolo11n_openvino_model",DEGRADE_ID="jpeg30")
    print("------------------------------------------------------------------------------")
    print("INT8")
    benchmark("models/yolo11n_int8_openvino_model",DEGRADE_ID="jpeg30")

    print(
        """
        Test-4: Downscale 
        .
        .
        .
        .
        .
        """
    )
    print("FP32")
    benchmark("models/yolo11n_openvino_model",DEGRADE_ID="downscale50")
    print("------------------------------------------------------------------------------")
    print("INT8")
    benchmark("models/yolo11n_int8_openvino_model",DEGRADE_ID="downscale50")

if __name__=="__main__":
    main()