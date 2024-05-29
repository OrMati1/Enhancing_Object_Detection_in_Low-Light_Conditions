import os
import subprocess
import torch
import torchvision

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

def run_yolov5_test(save_weights_path=None):
    print("Loading the YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Save the model weights if a path is provided
    if save_weights_path:
        torch.save(model.state_dict(), save_weights_path)
        print(f"Weights saved to {save_weights_path}")

    print("Running inference on a test image...")
    #img = 'https://ultralytics.com/images/zidane.jpg'
    img = '/test_img/3.bmp'
    results = model(img)

    print("Printing results...")
    results.print()

    print("Displaying results...")
    results.show()

if __name__ == '__main__':
    directory = '/FinalProject/YOLOv5'
    setup_yolov5(directory)

    print("Checking PyTorch and torchvision versions...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

    # Specify the path where you want to save the weights
    weights_save_path = '/YOLOv5/yolov5s.pt'
    run_yolov5_test(save_weights_path=weights_save_path)
