import os
import argparse
import torch
from omegaconf import OmegaConf

from trainer import train
from gnn import build_model
from dataset import get_dataset

parser = argparse.ArgumentParser(description="run experiements (note that all configs are stored in ./config, argparse only take the config file path)")
parser.add_argument("config", help="config .yaml path")



def train_embedding(config):    
    run_name = f"{config.dataset}_{config.gnn_model}_{config.suffix}"
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    print(f"Start training {run_name}")
    dataset, split = get_dataset(config)
    config.num_features = dataset.num_features
    config.num_classes = dataset.num_classes
    model = build_model(config)
    print("done loading model")
    
    train(model, dataset, split, config)
    torch.save(model.state_dict(), os.path.join(config.output_dir, run_name + ".pt"))


def main():
    config = OmegaConf.load(parser.parse_args().config)
    print(f"Running with following config: \n {OmegaConf.to_yaml(config)}")
    if config.task == "emb":
        train_embedding(config)


if __name__ == "__main__":
    main()










