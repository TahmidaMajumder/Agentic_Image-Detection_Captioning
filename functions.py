from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

def get_image_caption(image_path):
    """
    Generates a short caption for the provided image.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        str: A string representing the caption for the image.
    """
    image = Image.open(image_path).convert('RGB')
    
    model_name = "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load processor and model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Process image and generate caption
    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    
    # Decode the caption
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption

def detect_objects(image_path):
    """
    Detects objects in the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
    """
    image = Image.open(image_path).convert('RGB')

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections =""

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += "[{}, {}, {}, {}]".format(int(box[0]), int(box[1]), int(box[2]), int(box[3])) #bounding box
        detections += ' {}'.format(model.config.id2label[int(label)]) #labels
        detections += ' {}\n'.format(float(score)) #confidance score

    return detections

if __name__ == '__main__':
    image_path = r"O:\image_detection_caption_agent_based _system\test.png"
    detections = detect_objects(image_path)
    print(detections)