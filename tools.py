from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from PIL import Image
from typing import Literal
import torch


# Define input schema for Image Caption Tool
class ImageCaptionToolInput(BaseModel):
    img_path: str = Field(..., description="Path to the image file to be captioned")

# Define input schema for Object Detection Tool
class ObjectDetectionToolInput(BaseModel):
    img_path: str = Field(..., description="Path to the image file for object detection")


# captioner tool class
class ImageCaptionTool(BaseTool):
    name: str = "Image captioner"
    description: str = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )
    args_schema: Type[BaseModel] = ImageCaptionToolInput
    
    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')
        
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
    
    def _arun(self, img_path: str):
        raise NotImplementedError("This tool does not support async")


# detection tool class
class ObjectDetectionTool(BaseTool):
    name: str = "Object detector"
    description: str = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a list of all detected objects. Each element in the list in the format: "
        "[x1, y1, x2, y2] class_name confidence_score."
    )
    args_schema: Type[BaseModel] = ObjectDetectionToolInput
    
    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')
        
        # Load processor and model
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        
        # Process image and generate bounding box, label and confidence score
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        detections = ""
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += "[{}, {}, {}, {}]".format(int(box[0]), int(box[1]), int(box[2]), int(box[3])) #bounding box
            detections += ' {}'.format(model.config.id2label[int(label)]) #labels
            detections += ' {}\n'.format(float(score)) #confidance score
        
        return detections
    
    def _arun(self, img_path: str):
        raise NotImplementedError("This tool does not support async")