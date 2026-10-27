"""Reconstruct constituent four-vectors from a full binary Lund tree.

Trees are nested ``{"features": [...], "children": [...]}`` nodes (see
lund_tree_encoding.tokens_to_tree). Each split's features,
``[log(1/delta_R), log(z), varphi]``, describe it relative to its parent:
``varphi`` fixes the signed rapidity/azimuth separation of the harder and softer
branches and ``z`` their pT sharing. The jet's pT, pseudorapidity and azimuth are
given as ``root_kinematics`` and its mass follows from the splits, so the
constituents always sum exactly to the reconstructed jet.
"""

import math
import numpy as np

from lund_tree_encoding import pt_sharing


def _split_transverse(parent_pt, features):
    """Invert the transverse part of a split, independent of rapidity."""
    log_inv_dr, log_z, varphi = (float(value) for value in features[:3])
    if not all(math.isfinite(value) for value in (log_inv_dr, log_z, varphi)):
        raise ValueError("Split features must all be finite")

    delta_r = math.exp(-log_inv_dr)
    z = math.exp(log_z)
    if not 0 < z < 1:
        raise ValueError(f"z={z:g} is outside (0, 1)")


    # jet_full_declusterings uses varphi = atan2(y_hard - y_soft, hard.delta_phi_to(soft)),
    # and fastjet's delta_phi_to returns phi_soft - phi_hard, so negate to get hard - soft.
    angle = math.remainder(varphi, 2.0 * math.pi)
    delta_y = delta_r * math.sin(angle)
    delta_phi = -delta_r * math.cos(angle)
    # z = pt_soft / (pt_hard + pt_soft); the parent pT is the vector sum of the
    # daughter pT vectors, which fixes the scalar sum pt_hard + pt_soft.
    with np.errstate(divide="ignore"):
        scalar_pt = parent_pt * float(pt_sharing(delta_r, z, angle))
    if not math.isfinite(scalar_pt):
        raise ValueError("Back-to-back equal-pT split cannot carry the parent pT")
    return (1 - z) * scalar_pt, z * scalar_pt, delta_y, delta_phi


def _subtree_masses(node, parent_pt, masses):
    """Return a subtree's invariant mass from its parent pT, storing every node's mass in ``masses`` by id."""
    children = node["children"]
    mass = 0.0
    if children:
        pt_hard, pt_soft, delta_y, delta_phi = _split_transverse(parent_pt, node["features"])
        mt_hard = math.hypot(pt_hard, _subtree_masses(children[0], pt_hard, masses))
        mt_soft = math.hypot(pt_soft, _subtree_masses(children[1], pt_soft, masses))
        energy = mt_hard * math.cosh(delta_y) + mt_soft
        px = pt_hard * math.cos(delta_phi) + pt_soft
        py = pt_hard * math.sin(delta_phi)
        pz = mt_hard * math.sinh(delta_y)
        mass = math.sqrt(max(energy**2 - px**2 - py**2 - pz**2, 0.0))
    masses[id(node)] = mass
    return mass


def _split_parent(parent, features, mass_hard, mass_soft):
    """Invert one split into the two (possibly massive) daughter four-vectors."""
    energy, px, py, pz = (float(value) for value in parent)
    parent_pt = math.hypot(px, py)
    if parent_pt <= 0 or energy <= abs(pz):
        raise ValueError("Parent must have positive transverse momentum and finite rapidity")
    pt_hard, pt_soft, delta_y, delta_phi = _split_transverse(parent_pt, features)
    mt_hard = math.hypot(pt_hard, mass_hard)
    mt_soft = math.hypot(pt_soft, mass_soft)
    parent_plus = energy + pz
    parent_minus = energy - pz
    plus_factor = mt_hard * math.exp(delta_y) + mt_soft
    minus_factor = mt_hard * math.exp(-delta_y) + mt_soft
    rapidity_soft = 0.5 * (
        math.log(parent_plus / parent_minus) - math.log(plus_factor / minus_factor)
    )
    rapidity_hard = rapidity_soft + delta_y

    parent_phi = math.atan2(py, px)
    # Orient the daughter azimuths so their vector sum points along the parent pT.
    phi_soft_offset = math.atan2(
        -pt_hard * math.sin(delta_phi),
        pt_hard * math.cos(delta_phi) + pt_soft,
    )
    phi_soft = parent_phi + phi_soft_offset
    phi_hard = phi_soft + delta_phi

    def branch_p4(pt, mt, rapidity, phi):
        return np.asarray(
            [
                mt * math.cosh(rapidity),
                pt * math.cos(phi),
                pt * math.sin(phi),
                mt * math.sinh(rapidity),
            ],
            dtype=np.float64,
        )

    hard = branch_p4(pt_hard, mt_hard, rapidity_hard, phi_hard)
    soft = branch_p4(pt_soft, mt_soft, rapidity_soft, phi_soft)
    residual = hard + soft - np.asarray(parent, dtype=np.float64)
    scale = max(float(np.max(np.abs(parent))), 1.0)
    if np.max(np.abs(residual)) > 2e-6 * scale:
        raise ValueError(
            "Split coordinates are inconsistent with the supplied parent four-vector "
            f"(maximum closure residual {np.max(np.abs(residual)) / scale:.3g})"
        )
    return hard, soft


def root_four_vector(tree, root_kinematics, masses=None):
    """Return the jet [E, PX, PY, PZ] from its ``root_kinematics=(pt, eta, phi)``; the mass follows from the splits.

    ``masses`` collects every subtree mass by node id (see _subtree_masses).
    """
    if not tree["children"]:
        raise ValueError("A tree without splits carries no jet kinematics")
    pt, eta, phi = (float(value) for value in root_kinematics)
    if not (all(math.isfinite(value) for value in (pt, eta, phi)) and pt > 0):
        raise ValueError("Root kinematics must all be finite with a positive pT")
    mt = math.hypot(pt, _subtree_masses(tree, pt, {} if masses is None else masses))
    rapidity = math.asinh(pt * math.sinh(eta) / mt)  # pz = pt sinh(eta) = mt sinh(rapidity)
    return np.asarray(
        [mt * math.cosh(rapidity), pt * math.cos(phi), pt * math.sin(phi), mt * math.sinh(rapidity)],
        dtype=np.float64,
    )


def reconstruct_tree(tree, root_kinematics):
    """Return the jet four-vector and its leaf constituents in depth-first hard-first order.

    ``root_kinematics`` is the jet ``(pt, eta, phi)``.
    """
    masses = {}
    root_p4 = root_four_vector(tree, root_kinematics, masses)

    def visit(node, parent):
        children = node["children"]
        if not children:
            return [np.asarray(parent, dtype=np.float64)]
        hard, soft = _split_parent(parent, node["features"], masses[id(children[0])], masses[id(children[1])])
        return visit(children[0], hard) + visit(children[1], soft)

    constituents = visit(tree, root_p4)
    total = np.sum(constituents, axis=0)
    scale = max(float(np.max(np.abs(root_p4))), 1.0)
    if np.max(np.abs(total - root_p4)) > 1e-5 * scale:
        raise ValueError("Reconstructed constituents do not conserve the root four-vector")
    return root_p4, constituents

