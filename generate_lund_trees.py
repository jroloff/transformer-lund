#!/usr/bin/env python3
"""Generate Lund-tree jets with a trained train_unbinned.py --input_format lundtree model.

Each generated jet is conditioned on a jet from a tree file (by default the
model's validation file). The output is a samples .npz holding the generated
trees, the matching original trees and the jet conditioning. Load it with
lund_tree_encoding.load_lund_jets to make any plots, or remake the standard
validation plots with validate_unbinned.py --from-samples.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from helpers import set_seeds
from helpers_unbinned import load_checkpoint_model, load_torchpickle, lundtree_dataset
from helpers_plotting import generate_validation_samples, save_validation_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint from train_unbinned.py (e.g. checkpoints/best.pt)")
    parser.add_argument("--n-jets", type=int, default=10000, help="Number of jets to generate (at most the jets in --cond-file)")
    parser.add_argument("--cond-file", default=None, help="Tree HDF5 file whose jets condition the generation (default: the model's validation file)")
    parser.add_argument("--output", default=None, help="Output .npz (default: generated_jets.npz next to the checkpoint's run directory)")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"], help="Device (default: cuda if available, else cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Generation batch size (default: the training batch size)")
    parser.add_argument("--seed", type=int, default=0)
    cli = parser.parse_args()

    checkpoint = load_torchpickle(cli.checkpoint)
    args = argparse.Namespace(**checkpoint["args"])
    if args.input_format != "lundtree":
        parser.error(f"{cli.checkpoint} is a {args.input_format} model, not a lundtree model")
    args.device = cli.device
    args.validation_size = cli.n_jets
    set_seeds(cli.seed)

    dataset = lundtree_dataset(cli.cond_file or args.val_file, max_events=cli.n_jets)
    loader = DataLoader(dataset, batch_size=cli.batch_size or args.batch_size, shuffle=False)

    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(args, (1, *dataset.tokens.shape[1:]), (1, dataset.jet.shape[-1]), device, checkpoint_info=checkpoint)

    original, generated, jet_list, unavailable = generate_validation_samples([model], loader, args)
    output = cli.output or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cli.checkpoint))), "generated_jets.npz")
    save_validation_samples(output, original, generated, jet_list, ["original", "generated"], unavailable, args)


if __name__ == "__main__":
    main()
