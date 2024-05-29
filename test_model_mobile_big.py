import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import urllib.request
import logging
from datetime import datetime
from io import BytesIO
from mobie_model_big import MobileNetV3UNet

def load_image(image_path):
    """Load an image from a file path."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor
        transforms.Resize((1024, 1024))  # Resize to fit the model input, if necessary
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def save_output(image_tensor, output_path):
    """Save a tensor as an image."""
    image = transforms.ToPILImage()(image_tensor.squeeze(0))  # Remove batch dimension and convert to PIL
    image.save(output_path)

def load_yolov5_model(weights_path, device):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path).to(device)
    model.eval()
    return model

def draw_predictions(image, results, class_names):
    draw = ImageDraw.Draw(image)
    detections = results.pandas().xyxy[0]  # Convert results to pandas DataFrame
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = int(row['class'])
        class_name = class_names[cls] if cls < len(class_names) else str(cls)
        if conf > 0.2:  # Confidence threshold
            box_color = (255, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            draw.text((x1, y1), f'{class_name}, Conf: {conf:.2f}', fill=box_color)

def load_class_names(path):
    with open(path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def combine_images_with_title(original_image, enhanced_image):
    # Resize both images to the same size
    new_size = (max(original_image.width, enhanced_image.width), max(original_image.height, enhanced_image.height))
    original_image = original_image.resize(new_size)
    enhanced_image = enhanced_image.resize(new_size)

    # Create a new blank image with enough space for titles and images
    total_width = original_image.width + enhanced_image.width
    total_height = max(original_image.height, enhanced_image.height) + 50  # Add space for titles

    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Add original image
    combined_image.paste(original_image, (0, 50))

    # Add enhanced image
    combined_image.paste(enhanced_image, (original_image.width, 50))

    # Draw the title
    draw = ImageDraw.Draw(combined_image)
    title_font = ImageFont.truetype("arial.ttf", 24)
    title_text = "Comparative Object Detection: Original vs. Enhanced Images with YOLOv5"
    text_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text(((total_width - text_width) / 2, 10), title_text, fill=(0, 0, 0), font=title_font)

    return combined_image

def test_model(input_image_path, output_image_path, model_weights_path, yolov5_weights_path, class_names_path):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3UNet(num_classes=3).to(device)  # Use the new model

    # Load checkpoint
    checkpoint = torch.load(model_weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Load YOLOv5 model
    yolov5_model = load_yolov5_model(yolov5_weights_path, device)

    # Load class names
    class_names = load_class_names(class_names_path)

    # Load the image
    image = load_image(input_image_path).to(device)

    # Process the image
    with torch.no_grad():  # No need to calculate gradients
        enhanced_image = model(image)

    # Save the enhanced image
    save_output(enhanced_image, output_image_path)
    print(f"Enhanced image saved to {output_image_path}")

    # Convert tensor to PIL image for drawing
    enhanced_image_pil = transforms.ToPILImage()(enhanced_image.squeeze(0))

    # Perform YOLOv5 detection on the original image
    original_image_pil = Image.open(input_image_path).convert('RGB')
    original_results = yolov5_model(input_image_path)

    # Perform YOLOv5 detection on the enhanced image
    enhanced_results = yolov5_model(output_image_path)

    # Draw predictions on both images
    draw_predictions(original_image_pil, original_results, class_names)
    draw_predictions(enhanced_image_pil, enhanced_results, class_names)

    # Combine images with title
    combined_image = combine_images_with_title(original_image_pil, enhanced_image_pil)

    # Display results
    combined_image.show()

if __name__ == '__main__':
    # Define paths
    input_image_path = 'test_img/3.bmp'  # Path to test image
    output_image_path = 'results/enhanced_image.jpg'  # Path to save the enhanced image
    model_weights_path = 'model/mobile_big_epoch_96.pth'  # Path to the trained model weights
    yolov5_weights_path = 'yolov5s.pt'  # Path to the YOLOv5 model weights
    class_names_path = 'coco.names'  # Path to the class names file

    # Run the test
    test_model(input_image_path, output_image_path, model_weights_path, yolov5_weights_path, class_names_path)
