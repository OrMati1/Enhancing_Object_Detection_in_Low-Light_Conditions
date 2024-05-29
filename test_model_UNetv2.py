import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from UNetv2_model import UNetv2

# Load YOLOv5 model
def load_yolov5_model(weights_path, device):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path).to(device)
    model.eval()
    return model

# Load Image Enhancement model
def load_enhancement_model(weights_path, device):
    model = UNetv2()
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# Load class names
def load_class_names(path):
    with open(path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Enhance image
def enhance_image(image, model, device):
    transform = transforms.ToTensor()
    input_img = transform(image).unsqueeze(0).to(device)

    print(f"Input image size (to enhancement model): {input_img.shape}")

    with torch.no_grad():
        enhanced_img = model(input_img).squeeze(0).cpu()

    return transforms.ToPILImage()(enhanced_img)

# Draw predictions on image
def draw_predictions(image, results, class_names):
    draw = ImageDraw.Draw(image)
    detections = results.pandas().xyxy[0]  # Convert results to pandas DataFrame
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = int(row['class'])
        class_name = class_names[cls] if cls < len(class_names) else str(cls)
        if conf > 0.5:  # Confidence threshold
            print(f"Drawing box: ({x1, y1}, {x2, y2}), Conf: {conf}, Class: {class_name}")
            box_color = (255, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            draw.text((x1, y1), f'{class_name}, Conf: {conf:.2f}', fill=box_color)

def combine_images_with_title(original_image, enhanced_image):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to model weights and class names
    yolov5_weights_path = 'yolov5s.pt'
    class_names_path = 'coco.names'  # Path to your class names file
    enhancement_weights_path = 'model/unetv2_epoch_2.pth'

    # Load models
    yolov5_model = load_yolov5_model(yolov5_weights_path, device)
    enhancement_model = load_enhancement_model(enhancement_weights_path, device)

    # Load class names
    class_names = load_class_names(class_names_path)

    # Test image path
    input_image_path = 'test_img/4.png'

    # Process original image
    print("Processing original image with YOLOv5...")
    original_results = yolov5_model(input_image_path)
    original_image = Image.open(input_image_path).convert('RGB')

    # Enhance the image
    print("Enhancing the image...")
    enhanced_image = enhance_image(original_image, enhancement_model, device)

    # Save enhanced image temporarily
    enhanced_image_path = 'enhanced_img.jpg'
    enhanced_image.save(enhanced_image_path)

    # Process enhanced image with YOLOv5
    print("Processing enhanced image with YOLOv5...")
    enhanced_results = yolov5_model(enhanced_image_path)

    # Draw predictions on both images
    print("Drawing predictions on original image...")
    draw_predictions(original_image, original_results, class_names)
    print("Drawing predictions on enhanced image...")
    draw_predictions(enhanced_image, enhanced_results, class_names)

    # Combine images with title
    print("Combining images with title...")
    combined_image = combine_images_with_title(original_image, enhanced_image)

    # Display results
    combined_image.show()

    # Cleanup temporary file
    os.remove(enhanced_image_path)

if __name__ == '__main__':
    main()
