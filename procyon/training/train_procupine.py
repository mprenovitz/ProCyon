import argparse
from ProCyon.procyon.model.procupine_encoder import procupineVAE, mse_loss, nb_loss
import os
import torch
from torch.nn import functional as F
from datasets import load_dataset

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cvae", action="store_true")
    parser.add_argument("--load_from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", action="store_true")
    parser.add_argument("--loss_fn", default='mse')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=15)
    parser.add_argument("--input_size", type=int, default=28 * 28)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    return args
    
def train_vae(model, train_loader, args):
    optimizer = torch.optim.Adam(learning_rate=args.learning_rate)
    loss_fn = mse_loss if args.loss_fn != 'nb' else nb_loss
    loss = 0
    for x, y in train_loader:         
        oneHot = F.one_hot(y, 5000)
        x_pred, mu, logvar = model(x, oneHot)
        l = loss_fn(x_pred, x, mu, logvar)
        l.backward()
        optimizer.step()
        loss+=l
    return loss

def save_model(model, epoch):
    path = f"/checkpoints/procupine-chkpt{epoch}"
    try:
        os.makedirs(path, exist_ok=True)
        torch.save(model, path)
    except Exception:
        print(Exception, f"Unsuccessfully created directory at: {path}")

def forward(args):
    train_dataset = load_dataset("csv", "/datasets/5k_pbmc_protein.csv")

    model = procupineVAE(args.input_size, latent_size=args.latent_size)
    if args.load_from_checkpoint:
        model =  torch.load(args.checkpoint_path)

    for epoch_id in range(args.num_epochs):
        total_loss = train_vae(model, train_dataset, args)
        print(f"Epoch: {epoch_id} \tLoss: {total_loss / len(train_dataset):.6f}")
        if epoch_id+1 % 5 == 0:
            print(f"Saving checkpoint {epoch_id}")
            save_model(model, epoch_id)

    save_model(model, epoch_id)