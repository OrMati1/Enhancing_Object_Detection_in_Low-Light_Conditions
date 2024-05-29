import os
import subprocess

def setup_yolov5(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    os.chdir(directory)

    if not os.path.exists("yolov5"):
        print("Cloning the YOLOv5 repository...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)

    os.chdir("yolov5")

    print("Setting up YOLOv5 environment...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    weights_path = os.path.join("weights", "yolov5s.pt")
    if not os.path.exists(weights_path):
        print("Downloading YOLOv5 weights...")
        subprocess.run(["python", "path/to/yolov5/scripts/download_weights.py"], check=True)
    else:
        print("YOLOv5 weights already downloaded.")

if __name__ == '__main__':
    directory = '\FinalProject\YOLOv5'
    setup_yolov5(directory)
