import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def gram_schmidt(vectors):
    orthogonalized_vectors = []
    for vector in vectors:
        orthogonalized_vector = vector - sum(
            np.dot(vector, v) * v for v in orthogonalized_vectors
        )
        if np.linalg.norm(orthogonalized_vector) > 1e-10:
            orthogonalized_vectors.append(orthogonalized_vector)
    return orthogonalized_vectors


def calculate_spherical_coords(point, origin, ref1, ref2):
    # Calculate the displacement vectors
    displacement = point - origin
    print(f"Displacement vector: {displacement}")
    ref1_disp = ref1 - origin
    print(f"Reference 1 displacement: {ref1_disp}")
    ref2_disp = ref2 - origin
    print(f"Reference 2 displacement: {ref2_disp}")

    # Orthogonalize the basis vectors
    basis1 = ref1_disp / np.linalg.norm(ref1_disp)
    print(f"Basis 1 vector: {basis1}")
    orthogonal = ref2_disp - np.dot(ref2_disp, basis1) * basis1
    print(f"Orthogonalized vector: {orthogonal}")
    basis2 = orthogonal / np.linalg.norm(orthogonal)
    print(f"Basis 2 vector: {basis2}")
    normal = np.cross(basis1, basis2)
    print(f"Normal vector: {normal}")

    # Calculate spherical coordinates
    radius = np.linalg.norm(displacement)
    print(f"Radius: {radius}")
    polar = np.arccos(np.dot(displacement, normal) / radius)
    print(f"Polar angle: {polar}")
    proj = displacement - np.dot(displacement, normal) * normal
    print(f"Projection vector: {proj}")
    azimuthal = np.arccos(np.dot(proj, basis1) / np.linalg.norm(proj))
    print(f"Initial azimuthal angle: {azimuthal}")
    if np.dot(np.cross(basis1, proj), normal) < 0:
        azimuthal = -azimuthal
        print("Azimuthal angle flipped to negative")
    print(f"Final azimuthal angle: {azimuthal}")

    return radius, polar, azimuthal


def get_reference_atoms(molecule, atom_idx):
    """Get reference atoms for computing spherical coordinates."""
    xi = molecule.positions[atom_idx]

    # Get neighbor indices using ASE's get distance method
    neighbors = []
    cutoff = 2.0  # Typical bond length cutoff in Angstroms
    distances = molecule.get_all_distances()
    neighbor_indices = np.where(distances[atom_idx] < cutoff)[0]
    # exclude the atom itself
    neighbor_indices = [nbr_idx for nbr_idx in neighbor_indices if nbr_idx != atom_idx]

    for nbr_idx in neighbor_indices:
        # print(f"Found neighbor: {nbr_idx}, atom type: {molecule.symbols[nbr_idx]}")
        neighbors.append(nbr_idx)
    # print(f"Total neighbors: {len(neighbors)}")
    # print(f"{neighbors=}")

    if neighbors:
        f = neighbors[0]
        xf = molecule.positions[f]
    else:
        f = atom_idx
        xf = xi

    c1 = neighbors[1] if len(neighbors) > 1 else f
    c2 = neighbors[2] if len(neighbors) > 2 else c1

    return xi, xf, molecule.positions[c1], molecule.positions[c2]


def get_generation_descriptors(xi, xf, xc1, xc2):
    """Compute generation descriptors (spherical coordinates)."""
    di, theta_i, phi_i = calculate_spherical_coords(xi, xf, xc1, xc2)

    theta_i_norm = theta_i / np.pi
    phi_i_abs = np.abs(phi_i) / np.pi
    phi_i_sign = np.sign(phi_i)

    return [di, theta_i_norm, phi_i_abs, phi_i_sign]


def get_neighbor_vectors(coords, atom_idx):
    """Get the 4 nearest neighbor vectors, padded with zeros if needed."""
    distances = np.linalg.norm(coords - coords[atom_idx], axis=1)
    distances[atom_idx] = np.inf

    nearest = np.argsort(distances)[:4]
    vectors = []
    lengths = []

    for idx in nearest:
        lengths.append(distances[idx])
        vectors.append(coords[idx] - coords[atom_idx])

    # Pad to exactly 4 neighbors
    while len(vectors) < 4:
        lengths.append(0.0)
        vectors.append(np.zeros(3))

    return vectors, lengths


def get_bond_angles(neighbor_vecs):
    """Compute bond angles between all pairs of neighbor vectors."""
    angles = []
    for k in range(3):
        for l in range(k + 1, 4):
            vec1, vec2 = neighbor_vecs[k], neighbor_vecs[l]
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta)
            else:
                angle = 0.0
            angles.append(angle)

    return angles


def compute_descriptors(molecule):
    """Compute descriptors for each atom in the molecule."""
    coords = molecule.GetConformer().GetPositions()
    descriptors = []

    for i in range(molecule.GetNumAtoms()):
        # Get reference atoms for spherical coordinates
        xi, xf, xc1, xc2 = get_reference_atoms(molecule, i, coords)

        # Compute generation descriptors
        g_i = get_generation_descriptors(xi, xf, xc1, xc2)

        # Get neighbor vectors and distances
        neighbor_vecs, lj_list = get_neighbor_vectors(coords, i)

        # Compute bond angles
        alpha_list = get_bond_angles(neighbor_vecs)

        # Normalize descriptors
        lj_list = [np.log(lj + 1e-8) for lj in lj_list]
        alpha_list = [alpha / np.pi for alpha in alpha_list]

        # Combine all descriptors
        u_i = lj_list + alpha_list
        z_i = np.concatenate([g_i, u_i])
        descriptors.append(z_i)

    return np.array(descriptors)


if __name__ == "__main__":
    pass
