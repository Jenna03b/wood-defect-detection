from ultralytics import YOLO

def export_to_onnx():
    print("Loading PyTorch model...")
    model = YOLO("weights/best.pt")

    print("Exporting to ONNX format...")
    model.export(format="onnx", dynamic=True, imgsz=768)
    print("Export successful! File saved to weights/best.onnx")

if __name__ == "__main__":
    export_to_onnx()