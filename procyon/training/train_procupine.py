import argparse
from procyon.model.procupine_encoder import procupineVAE
import os
import torch
from torch.nn import functional as F
from datasets import load_dataset

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from_checkpoint", default=False, action="store_true")
    parser.add_argument("--loss_fn", default='mse')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", help="Save model checkpoints here")
    parser.add_argument("--data_dir", help="Path to training data")
    args = parser.parse_args()
    return args
    
def train_vae(model, train_loader, args, optimizer,epoch, device):
    loss = 0
    for y in train_loader:  
        y = torch.stack(list(y.values()), dim=1).to(device)
        y_pred, mu, logvar = model(y)
        l = 0
        if args.loss_fn == 'nb':
            l = model.nb_loss(y=y,y_pred=y_pred, mu=mu, logvar=logvar)
        else:
            l = model.mse_loss(y=y,y_pred=y_pred, mu=mu, logvar=logvar)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss+=l
    return loss

def save_model(model, epoch, args):
    path = args.output_dir
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            print(Exception, f"Unsuccessfully created directory at: {path}")
    torch.save(model, os.path.join(path, f"procupine-chkpt{epoch}.pth"))

def forward(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = load_dataset("csv", data_files=args.data_dir)['train'].remove_columns("Unnamed: 0")
    train_dataset = train_dataset.with_format("torch", device=device)
    input_size = len(train_dataset.column_names)
    
    train_dataset = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    model = procupineVAE(input_size, args.hidden_size, args.latent_size).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    if args.load_from_checkpoint:
        model =  torch.load(args.checkpoint_path)

    for epoch_id in range(args.num_epochs):
        total_loss = train_vae(model, train_dataset, args, optimizer,epoch_id, device)
        print(f"Epoch: {epoch_id} \tLoss: {total_loss / len(train_dataset):.6f}")
        if epoch_id+1 % 5 == 0:
            print(f"Saving checkpoint {epoch_id}")
            save_model(model, epoch_id)

    save_model(model, epoch_id)
if __name__ == "__main__":
    args = parseArguments()
    forward(args)