'''Checks that full Lund trees reconstruct the original jet and constituent four-vectors.

Builds trees from ROOT constituents, inverts them with reconstruct_lund_tree.py,
and compares against the jet of the same constituents.
'''

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from ljpHelpers import constituents_to_lund_tree
from lund_tree_encoding import nodes_to_tree
from reconstruct_lund_tree import reconstruct_tree

BLUE, ORANGE = "#2a78d6", "#eb6834"


def kinematics(p4):
    """Return pT, rapidity, phi and mass for ``[..., 4]`` arrays ordered [E, px, py, pz]."""
    e, px, py, pz = np.moveaxis(np.asarray(p4, dtype=np.float64), -1, 0)
    pt = np.hypot(px, py)
    return pt, 0.5 * np.log((e + pz) / (e - pz)), np.arctan2(py, px), np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0, None))


def collect(root_file, radius):
    tree = uproot.open(root_file)["tree"]
    arrays = tree.arrays(["part_energy", "part_px", "part_py", "part_pz"], library="np")
    results = {"jets": [], "jets_reco": [], "leaves": [], "leaves_reco": []}
    failures = 0
    for index in range(len(arrays["part_energy"])):
        vectors = np.column_stack(
            [arrays[name][index] for name in ("part_energy", "part_px", "part_py", "part_pz")]
        ).astype(np.float64)
        # The original jet and its C/A leaves, in the same hard-first preorder as the tree
        leaves = []
        nodes, jet = constituents_to_lund_tree(vectors, radius=radius, return_jet=True, leaves=leaves)
        if len(nodes) < 3:
            continue  # no jet, or a single-constituent jet without a split
        try:
            # The tree fixes everything but the jet's pT and direction, which the generation is conditioned on
            pt, _, phi, _ = kinematics(jet)
            jet_reco, leaves_reco = reconstruct_tree(nodes_to_tree(nodes), (pt, np.arcsinh(jet[3] / pt), phi))
        except ValueError as exc:
            print(f"Event {index} failed to reconstruct: {exc}")
            failures += 1
            continue
        results["jets"].append(jet)
        results["jets_reco"].append(jet_reco)
        results["leaves"].append(np.asarray(leaves))
        results["leaves_reco"].append(np.asarray(leaves_reco))
    for key in ("jets", "jets_reco"):
        results[key] = np.asarray(results[key])
    for key in ("leaves", "leaves_reco"):
        results[key] = np.concatenate(results[key])
    return results, failures


def log10_abs(values):
    return np.log10(np.clip(np.abs(values), 1e-16, None))


def style(axis, xlabel, ylabel):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(color="#e4e3df", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot(results, output):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    jet = kinematics(results["jets"])
    jet_reco = kinematics(results["jets_reco"])
    leaf = kinematics(results["leaves"])
    leaf_reco = kinematics(results["leaves_reco"])
    residual_bins = np.linspace(-16, 0, 65)

    # (a) Jet mass spectrum: original vs reconstructed.
    axis = axes[0, 0]
    bins = np.linspace(0, np.percentile(jet[3], 99.5), 50)
    axis.hist(jet[3], bins=bins, histtype="step", linewidth=2, color=BLUE, label="Original anti-kT jet")
    axis.hist(jet_reco[3], bins=bins, histtype="step", linewidth=2, linestyle="--", color=ORANGE, label="Reconstructed from tree")
    axis.set_title("Jet mass")
    axis.legend(frameon=False)
    style(axis, "Jet mass [GeV]", "Jets")

    # (b, c) Jet-level relative residuals.
    for axis, (quantity, index) in zip(axes[0, 1:], (("pT", 0), ("mass", 3))):
        axis.hist(log10_abs(jet_reco[index] / jet[index] - 1), bins=residual_bins, histtype="step", linewidth=2, color=BLUE)
        axis.set_title(f"Jet {quantity}: relative residual")
        style(axis, rf"$\log_{{10}}|{quantity}_{{reco}}/{quantity}_{{true}} - 1|$", "Jets")

    # (d) Constituent pT, original vs reconstructed, matched in tree order.
    axis = axes[1, 0]
    axis.scatter(leaf[0], leaf_reco[0], s=8, color=BLUE, alpha=0.4, linewidths=0)
    limits = [max(leaf[0].min(), 1e-2), leaf[0].max()]
    axis.plot(limits, limits, color="#6b6a63", linewidth=1, linestyle=":", label="y = x")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title("Constituent pT")
    axis.legend(frameon=False, loc="upper left")
    style(axis, "Original constituent pT [GeV]", "Reconstructed constituent pT [GeV]")

    # (e) Constituent energy relative residual.
    axis = axes[1, 1]
    energy_residual = results["leaves_reco"][:, 0] / results["leaves"][:, 0] - 1
    axis.hist(log10_abs(energy_residual), bins=residual_bins, histtype="step", linewidth=2, color=BLUE)
    axis.set_title("Constituent energy: relative residual")
    style(axis, r"$\log_{10}|E_{reco}/E_{true} - 1|$", "Constituents")

    # (f) Angular distance between each original and reconstructed constituent.
    axis = axes[1, 2]
    delta_r = np.hypot(leaf_reco[1] - leaf[1], np.remainder(leaf_reco[2] - leaf[2] + np.pi, 2 * np.pi) - np.pi)
    axis.hist(log10_abs(delta_r), bins=residual_bins, histtype="step", linewidth=2, color=BLUE)
    axis.set_title(r"Constituent direction: $\Delta R$(reco, original)")
    style(axis, r"$\log_{10}\Delta R$", "Constituents")

    fig.suptitle(f"Lund-tree reconstruction vs original jets ({len(jet[0]):,} jets, massless constituents)", fontsize=14)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved validation plots: {output}")


def summarize(results):
    pt, _, _, mass = kinematics(results["jets"])
    pt_reco, _, _, mass_reco = kinematics(results["jets_reco"])
    energy_residual = np.abs(results["leaves_reco"][:, 0] / results["leaves"][:, 0] - 1)
    p4_residual = np.abs(results["jets_reco"] - results["jets"]).max(axis=1) / results["jets"][:, 0]
    print(f"{len(pt)} jets, {len(energy_residual)} constituents")
    for name, values in (
        ("jet four-vector max|diff|/E", p4_residual),
        ("jet pT relative", np.abs(pt_reco / pt - 1)),
        ("jet mass relative", np.abs(mass_reco / mass - 1)),
        ("constituent E relative", energy_residual),
    ):
        print(f"  {name:30s} median {np.median(values):.2e}  99% {np.percentile(values, 99):.2e}  max {values.max():.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", help="JetClass ROOT file with part_{energy,px,py,pz}")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--output", default="plots/lund_reconstruction_validation.png")
    args = parser.parse_args()

    results, failures = collect(args.root_file, args.radius)
    print(f"Reconstruction failures: {failures}")
    summarize(results)
    plot(results, args.output)


if __name__ == "__main__":
    main()
