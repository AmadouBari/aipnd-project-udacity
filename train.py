import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
import os

def get_input_args():
    """
    Retrieves and parses command line arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13',
                       help='Model architecture (default: vgg13)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--hidden_units', type=int, default=512,
                       help='Number of hidden units (default: 512)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    
    return parser.parse_args()

def load_data(data_dir):
    """
    Loads and transforms the data
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load the datasets
    image_datasets = {
        'train': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train']),
        'valid': datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid'])
    }
    
    # Create the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)
    }
    
    return dataloaders, image_datasets

def build_model(arch, hidden_units):
    """
    Builds the model with specified architecture
    """
    if arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        input_size = 25088
    elif arch == 'vgg13':
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        input_size = 25088
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Define new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    return model, input_size

def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    """
    Trains the model
    """
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 40
    
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

def main():
    args = get_input_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load data
    dataloaders, image_datasets = load_data(args.data_directory)
    
    # Build model
    model, input_size = build_model(args.arch, args.hidden_units)
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Train model
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)
    
    # Save checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'input_size': input_size,
        'output_size': 102,
        'hidden_layers': [args.hidden_units],
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs,
        'arch': args.arch  # Add this line
    }
    
    torch.save(checkpoint, f"{args.save_dir}/checkpoint.pth")
    print(f"Model saved to {args.save_dir}/checkpoint.pth")

if __name__ == '__main__':
    main()
