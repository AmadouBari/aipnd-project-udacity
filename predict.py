import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import json
import os

def get_input_args():
    """
    Retrieves and parses command line arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=1,
                       help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                       help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for inference')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    """
    Loads and rebuilds a trained model from checkpoint
    """
    checkpoint = torch.load(filepath, weights_only=True)
    
    # Default to vgg13 if architecture is not specified in checkpoint
    arch = checkpoint.get('arch', 'vgg13')
    
    if arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:  # vgg13
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier
    classifier = torch.nn.Sequential(
        torch.nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0]),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(checkpoint['hidden_layers'][0], checkpoint['output_size']),
        torch.nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model
    """
    # Open the image
    img = Image.open(image_path)
    
    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transformations
    img_tensor = preprocess(img)
    
    return img_tensor

def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model
    """
    # Process image
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Set model to evaluation mode and move to device
    model.to(device)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_class.cpu().numpy()[0]]  # Move to CPU before converting to NumPy
    
    return top_p.cpu().numpy()[0], top_classes  # Move to CPU before converting to NumPy

def main():
    # Get command line arguments
    args = get_input_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Load model from checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Make prediction
    probs, classes = predict(args.input, model, device, args.top_k)
    
    # Convert class indices to flower names
    flower_names = [cat_to_name[cls] for cls in classes]
    
    # Print results
    print("\nPredictions:")
    for i in range(len(probs)):
        print(f"{flower_names[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
  
    main()
