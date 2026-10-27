"""Model-space encoding of Lund-tree node features and token-sequence helpers.

Token sequences are depth-first preorder nodes of shape ``[..., n_nodes, n_features + 1]``: node features
followed by a split flag that is 1 for a split, 0 for a leaf and the pad value for padding. Model tokens hold
the encoded features (and optionally the input-only context), physical tokens the decoded ones.
"""

import numpy as np

from collections import namedtuple

# The features describing a split relative to its node. With the jet direction
# and pT given, they determine every constituent four-vector.
SPLIT_FEATURE_NAMES = ("log_inv_delta_R", "log_z", "varphi")

# Column of each feature in node features and tokens
DR, Z, VARPHI = range(len(SPLIT_FEATURE_NAMES))

# Plot labels of each physical node feature, in column order
FEATURE_LABELS = [r"$\log(1/\Delta R)$", r"$\log(z)$", r"$\varphi$"]


def pt_sharing(delta_r, z, varphi):
    """Return ``(pt_hard + pt_soft) / pt_parent`` of a split.

    The daughter pT vectors sum to the parent pT and are ``delta_phi = -delta_R
    cos(varphi)`` apart in azimuth (varphi = atan2(y_hard - y_soft, phi_soft -
    phi_hard)), with ``z`` the soft fraction
    of their scalar sum. Then ``pt_soft = z * sharing * pt_parent`` and
    ``kt = pt_soft * sin(delta_R)``.
    """
    delta_phi = -delta_r * np.cos(varphi)
    return 1.0 / np.sqrt((1 - z) ** 2 + z**2 + 2 * z * (1 - z) * np.cos(delta_phi))


# Physical bounds on the split features. The model learns and samples them in an
# unbounded space, so every generated split decodes to a kinematically valid one.
DELTA_R_MAX = 1.0  # C/A reclustering radius in ljpHelpers.jet_full_declusterings
#Technically, this can be larger than the jet radius, but since we are using a jet
# radius of R=0.8, this gives a bit of wiggle room. It might be better to enforce
# a slightly sharper boundary, but it should be a bit larger than the jet radius.

Z_MAX = 0.5  # the softer branch never carries more than half the scalar pT

# The model also builds in the C/A angular ordering. With E-scheme recombination the merged axis lies between the
# two merged objects, so the next merge can be up to sqrt(3)/2 closer than the last one: a split's delta_R is
# bounded by DELTA_R_RATIO_MAX times its parent's delta_R.
DELTA_R_RATIO_MAX = 2.0 / np.sqrt(3.0)


def encode_features(features, delta_r_max=DELTA_R_MAX):
    """Map physical internal-node features to the model's unbounded space.

    ``log(1/delta_R)`` becomes ``softplus^-1(log(delta_r_max/delta_R))`` and
    ``log(z)`` becomes ``softplus^-1(log(Z_MAX/z))``; varphi passes through.
    ``delta_r_max`` is a scalar or a per-row array (see encode_trees).
    """
    encoded = np.array(features, dtype=np.float64)
    for column, gap in ((DR, encoded[..., DR] + np.log(delta_r_max)), (Z, np.log(Z_MAX) - encoded[..., Z])):
        gap = np.maximum(gap, 1e-6)
        encoded[..., column] = gap + np.log(-np.expm1(-gap))
    return encoded


def decode_features(encoded, delta_r_max=DELTA_R_MAX):
    """Inverse of encode_features; any real input decodes inside the physical bounds."""
    features = np.array(encoded, dtype=np.float64)
    features[..., DR] = np.logaddexp(0.0, features[..., DR]) - np.log(delta_r_max)
    features[..., Z] = np.log(Z_MAX) - np.logaddexp(0.0, features[..., Z])
    return features


# A node met while walking a tree: its depth, whether it is the soft child, its pT,
# whether it lies on the primary Lund plane (the hard branch from the root) and its delta_R bound.
TreeNode = namedtuple("TreeNode", ["depth", "is_soft", "pt", "primary", "delta_r_bound"])

# The physical description of a split: delta_R, z and varphi.
Split = namedtuple("Split", ["delta_r", "z", "varphi"])

# Input-only context of each node: its depth, whether it is the soft child, and its pT relative to the jet.
CONTEXT_NAMES = ("log1p_depth", "is_soft_child", "log_pt_fraction")


class TreeWalker:
    """Follows a depth-first preorder tree node by node.

    ``step`` consumes the next node and returns it as a TreeNode plus, for a
    split, its Split. Node pTs follow down the tree from ``root_pt`` through
    pt_sharing. Split features are physical, or with ``encoded`` in model space,
    where each delta_R is bounded by DELTA_R_RATIO_MAX times its parent's delta_R
    (C/A angular ordering; the root by DELTA_R_MAX). The primary plane follows the hard branch from the root.
    ``context`` gives the context (CONTEXT_NAMES) of the next node, with pTs relative to ``root_pt``.
    """

    def __init__(self, root_pt=1.0, encoded=False):
        self.encoded = encoded
        self.stack = [TreeNode(0, False, float(root_pt), True, DELTA_R_MAX)]  # open child slots, next on top

    def next_node(self):
        """The node the next step consumes, or None once the tree is complete."""
        return self.stack[-1] if self.stack else None

    def context(self):
        """The context of the next node, or zeros once the tree is complete."""
        node = self.next_node()
        if node is None:
            return np.zeros(len(CONTEXT_NAMES))
        return np.array([np.log1p(node.depth), float(node.is_soft), np.log(node.pt)])

    def step(self, features, split):
        node = self.stack.pop()
        if not split:
            return node, None
        physical = decode_features(features, node.delta_r_bound) if self.encoded else features
        delta_r, z, varphi = np.exp(-physical[DR]), np.exp(physical[Z]), physical[VARPHI]
        scalar_pt = node.pt * pt_sharing(delta_r, z, varphi)
        child_bound = min(DELTA_R_MAX, DELTA_R_RATIO_MAX * delta_r)
        self.stack.append(TreeNode(node.depth + 1, True, z * scalar_pt, False, child_bound))
        self.stack.append(TreeNode(node.depth + 1, False, (1 - z) * scalar_pt, node.primary, child_bound))
        return node, Split(delta_r, z, varphi)


def walk_tree(tokens, root_pt=1.0, encoded=True):
    """Yield ``(index, node, split)`` for every node of one ``[n_nodes, F + 1]`` token sequence (see TreeWalker)."""
    walker = TreeWalker(root_pt, encoded)
    for index, token in enumerate(tokens):
        if walker.next_node() is None or token[-1] not in (0, 1):
            return
        node, split = walker.step(token[:-1], token[-1] == 1)
        yield index, node, split


def encode_trees(features, splits):
    """Encode physical ``[N, n_nodes, F]`` trees in one walk over each tree.

    Returns the model-space features, with each split's delta_R bounded relative to its parent's (see TreeWalker;
    every C/A split with E-scheme recombination satisfies the bound), and the ``[N, n_nodes, len(CONTEXT_NAMES)]``
    context of node k+1 at row k (zeros past the end).
    """
    features = np.array(features, dtype=np.float64)
    bounds = np.full(splits.shape, DELTA_R_MAX)
    context = np.zeros(features.shape[:2] + (len(CONTEXT_NAMES),))
    for jet in range(len(features)):
        walker = TreeWalker()
        for index, split in enumerate(splits[jet]):
            if walker.next_node() is None or split not in (0, 1):
                break
            bounds[jet, index] = walker.next_node().delta_r_bound
            walker.step(features[jet, index], split == 1)
            context[jet, index] = walker.context()
    encoded = np.array(features, dtype=np.float64)
    internal = splits == 1
    encoded[internal] = encode_features(features[internal], bounds[internal])
    return encoded, context


def physical_tokens(tokens):
    """Decode model tokens to physical ``[N, n_nodes, F + 1]`` tokens for plotting and reconstruction.

    Drops the input-only context columns and decodes each split with its parent's delta_R bound. Leaves keep zero
    features and padding the pad value.
    """
    tokens = np.concatenate([tokens[..., :len(SPLIT_FEATURE_NAMES)], tokens[..., -1:]], axis=-1).astype(np.float64)
    for jet in range(len(tokens)):
        for index, _, split in walk_tree(tokens[jet]):
            if split is not None:
                tokens[jet, index, [DR, Z, VARPHI]] = [-np.log(split.delta_r), np.log(split.z), split.varphi]
    return tokens


def nodes_to_tree(nodes):
    """Nest preorder ``(features, split)`` nodes into ``{"features": [...], "children": [...]}`` (see reconstruct_lund_tree)."""
    cursor = iter(nodes)

    def build():
        features, split = next(cursor)
        return {"features": [float(value) for value in features], "children": [build(), build()] if split else []}

    return build()


def tokens_to_tree(tokens):
    """Nest one physical ``[n_nodes, n_features + 1]`` token sequence, or None when it is not a complete binary tree."""
    nodes = []
    pending = 1
    for row in np.asarray(tokens, dtype=np.float64):
        if not pending:
            break
        split = row[-1] == 1
        nodes.append((row[:-1], split))
        pending += 1 if split else -1
    return None if pending else nodes_to_tree(nodes)


def lund_plane_points(tokens, root_pt):
    """Return ``[log(kt), log(1/delta_R)]`` for every split on the primary Lund plane.

    ``tokens`` are physical ``[N, n_nodes, F + 1]`` tokens and ``root_pt`` the ``[N]`` jet pTs.
    """
    points = []
    for row, jet_pt in zip(tokens, np.asarray(root_pt, dtype=np.float64)):
        for _, node, split in walk_tree(row, jet_pt, encoded=False):
            if split is not None and node.primary:
                kt = split.z * node.pt * pt_sharing(*split) * np.sin(split.delta_r)
                points.append((np.log(kt), -np.log(split.delta_r)))
    return np.array(points).reshape(-1, 2)


def tokens_to_constituents(tokens, n_constituents, root_kinematics, pad_value=-1):
    """Reconstruct ``[N, n_constituents, 4]`` constituent ``[E, px, py, pz]`` from physical ``[N, n_nodes, F + 1]`` tokens.

    ``root_kinematics`` gives each jet's ``(pt, eta, phi)``; its mass follows
    from the tree. Unused slots, and every slot of a jet that cannot be
    reconstructed, are set to ``pad_value``. Returns the constituents and the
    number of failed jets.
    """
    from reconstruct_lund_tree import reconstruct_tree

    constituents = np.full((len(tokens), n_constituents, 4), float(pad_value))
    failures = 0
    for index, jet_tokens in enumerate(tokens):
        tree = tokens_to_tree(jet_tokens)
        try:
            if tree is None:
                raise ValueError("incomplete tree")
            _, leaves = reconstruct_tree(tree, root_kinematics[index])
        except ValueError:
            failures += 1
            continue
        leaves = np.asarray(leaves)[:n_constituents]
        constituents[index, : len(leaves)] = leaves
    return constituents, failures


def summarize_trees(tokens, root_kinematics, reconstruct=True):
    """Describe physical ``[N, n_nodes, F + 1]`` tokens of jets with ``[N, 3]`` ``root_kinematics`` (pt, eta, phi).

    Returns a dict of
      ``tokens``         the physical tokens
      ``split``          [N, n_nodes] split flags (1 split, 0 leaf, -1 padding)
      ``features``       [N, n_nodes, F] physical split features (log(1/delta_R), log(z), varphi), NaN for non-splits
      ``primary``        [M, 2] primary Lund-plane points, see lund_plane_points
      ``constituents``   [N, n_nodes//2 + 1, 4] reconstructed [E, px, py, pz], padded with -1 (with ``reconstruct``)
      ``failed``         the number of jets that could not be reconstructed (with ``reconstruct``)
    """
    split = tokens[..., -1]
    summary = {
        "tokens": tokens,
        "split": split,
        "features": np.where((split == 1)[..., None], tokens[..., :-1], np.nan),
        "primary": lund_plane_points(tokens, root_kinematics[:, 0]),
    }
    if reconstruct:
        summary["constituents"], summary["failed"] = tokens_to_constituents(tokens, tokens.shape[1] // 2 + 1, root_kinematics)
    return summary


def load_lund_jets(path, reconstruct=True):
    """Load a samples file (validation_samples.npz or generate_lund_trees.py output) into plain arrays.

    Returns ``{label: sample}`` for the original jets and each generated set, where every sample is the
    summarize_trees dict plus ``jets``, the [N, 3] jet conditioning [pt, eta, phi].
    """
    data = np.load(path, allow_pickle=True)
    jets = data["jet_list"]
    root = np.asarray(jets, dtype=np.float64)[:, :3]
    labels = data["labels"].tolist()

    arrays = {labels[0]: data["original"]}
    for i, label in enumerate(labels[1:]):
        if f"generated_{i}" in data:
            arrays[label] = data[f"generated_{i}"]

    return {label: {**summarize_trees(physical_tokens(tokens), root, reconstruct), "jets": jets} for label, tokens in arrays.items()}
