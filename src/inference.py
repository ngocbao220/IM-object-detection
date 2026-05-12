import torch
import cv2
import numpy as np
from model import get_model
from constants import NUM_CLASSES, IDX_TO_CLASS
from PIL import Image
import torchvision.transforms as T

def get_prediction(image_path, model, device, threshold):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model([img_tensor])
        
    pred_score = list(prediction[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    
    if len(pred_t) == 0:
        return None, None, None
        
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in prediction[0]['boxes'].detach().cpu().numpy()]
    pred_class = [IDX_TO_CLASS[i] for i in prediction[0]['labels'].detach().cpu().numpy()]
    
    pred_boxes = pred_boxes[:len(pred_t)]
    pred_class = pred_class[:len(pred_t)]
    pred_score = pred_score[:len(pred_t)]
    
    return pred_boxes, pred_class, pred_score

def detect_objects(image_path, model_path, threshold=0.5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    boxes, pred_cls, scores = get_prediction(image_path, model, device, threshold)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if boxes is not None:
        for i in range(len(boxes)):
            cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), 
                          (int(boxes[i][1][0]), int(boxes[i][1][1])), 
                          color=(0, 255, 0), thickness=2)
            cv2.putText(img, f"{pred_cls[i]}: {scores[i]:.2f}", 
                        (int(boxes[i][0][0]), int(boxes[i][0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
    
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/inference.py <image_path> <model_path>")
    else:
        detect_objects(sys.argv[1], sys.argv[2])
