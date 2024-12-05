import numpy as np
import pytest
from src.mol_struc_tok import (
    gram_schmidt,
    calculate_spherical_coords,
    get_reference_atoms,
    get_generation_descriptors,
    get_neighbor_vectors,
    get_bond_angles,
    compute_descriptors,
)
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read


@pytest.fixture
def water_xyz():
    return """3
Water molecule
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116"""


def test_gram_schmidt():
    # Test orthogonalization of 2D vectors
    vectors = [np.array([1.0, 0.0]), np.array([1.0, 1.0])]
    result = gram_schmidt(vectors)
    assert len(result) == 2
    # Check vectors are orthogonal
    dot_product = np.dot(result[0], result[1])
    assert abs(dot_product) < 1e-10

    # Test with linearly dependent vectors
    vectors = [np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    result = gram_schmidt(vectors)
    assert len(result) == 1


def test_calculate_spherical_coords():
    # Test case 1: Point on x-axis
    point = np.array([1.0, 0.0, 0.0])
    origin = np.array([0.0, 0.0, 0.0])
    ref1 = np.array([1.0, 0.0, 0.0])
    ref2 = np.array([0.0, 0.0, 1.0])

    radius, polar, azimuthal = calculate_spherical_coords(point, origin, ref1, ref2)
    assert radius == pytest.approx(1.0)
    assert polar == pytest.approx(np.pi / 2)
    assert azimuthal == pytest.approx(0.0)

    # Test case 2: Point on z-axis
    point = np.array([0.0, 0.0, 1.0])
    radius, polar, azimuthal = calculate_spherical_coords(point, origin, ref1, ref2)
    assert radius == pytest.approx(1.0)
    assert polar == pytest.approx(0.0)  # Point aligned with normal vector
    assert azimuthal == pytest.approx(np.pi / 2)

    # Test case 3: Point in x-y plane
    point = np.array([1.0, 1.0, 0.0])
    radius, polar, azimuthal = calculate_spherical_coords(point, origin, ref1, ref2)
    assert radius == pytest.approx(np.sqrt(2.0))
    assert polar == pytest.approx(np.pi / 2)
    assert azimuthal == pytest.approx(np.pi / 4)

    # Test case 4: Non-zero origin
    point = np.array([2.0, 0.0, 0.0])
    origin = np.array([1.0, 0.0, 0.0])
    radius, polar, azimuthal = calculate_spherical_coords(point, origin, ref1, ref2)
    assert radius == pytest.approx(1.0)
    assert polar == pytest.approx(np.pi / 2)
    assert azimuthal == pytest.approx(0.0)


def test_gram_schmidt_3d():
    # Test orthogonalization of 3D vectors
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    ]
    result = gram_schmidt(vectors)
    assert len(result) == 3

    # Check all vectors are orthogonal to each other
    assert abs(np.dot(result[0], result[1])) < 1e-10
    assert abs(np.dot(result[1], result[2])) < 1e-10
    assert abs(np.dot(result[0], result[2])) < 1e-10


@pytest.mark.parametrize(
    "xyz_file, coordinates, ids",
    [
        (
            "tests/water.xyz",
            np.array(
                [
                    [0.0, 0.0, 0.11779],
                    [0.0, 0.75545, -0.47116],
                    [0.0, -0.75545, -0.47116],
                ]
            ),
            [0, 1, 2, 2],
        ),
        (
            "tests/methane.xyz",
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.089],
                    [1.02672, 0.0, -0.363],
                    [-0.51336, -0.88916, -0.363],
                    [-0.51336, 0.88916, -0.363],
                ]
            ),
            [0, 1, 2, 3],
        ),
    ],
)
def test_get_reference_atoms(xyz_file, coordinates, ids):
    # Create a simple molecule for testing
    mol = read(xyz_file, format="xyz")

    xi, xf, xc1, xc2 = get_reference_atoms(mol, 0)
    print(f"{xi=}, {xf=}, {xc1=}, {xc2=}")
    assert xi == pytest.approx(coordinates[ids[0]])
    assert xf == pytest.approx(coordinates[ids[1]])
    assert xc1 == pytest.approx(coordinates[ids[2]])
    assert xc2 == pytest.approx(coordinates[ids[3]])

    # # Test hydrogen (should use fallbacks)
    # xi, xf, xc1, xc2 = get_reference_atoms(mol, 1)
    # print(f"{xi=}, {xf=}, {xc1=}, {xc2=}")
    # assert xi == pytest.approx(coordinates[1])
    # assert xf == pytest.approx(coordinates[0])
    # assert xc1 == pytest.approx(coordinates[1])
    # assert xc2 == pytest.approx(coordinates[2])


def test_get_generation_descriptors():
    # Test with known geometry
    xi = np.array([0.0, 0.0, 0.0])
    xf = np.array([1.0, 0.0, 0.0])
    xc1 = np.array([0.0, 1.0, 0.0])
    xc2 = np.array([0.0, 0.0, 1.0])

    descriptors = get_generation_descriptors(xi, xf, xc1, xc2)
    assert len(descriptors) == 4
    assert descriptors[0] == pytest.approx(1.0)  # distance
    assert 0 <= descriptors[1] <= 1  # normalized theta
    assert 0 <= descriptors[2] <= 1  # normalized phi abs
    assert descriptors[3] in [-1, 0, 1]  # phi sign


def test_get_neighbor_vectors():
    # Create test coordinates
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )

    vectors, lengths = get_neighbor_vectors(coords, 0)
    assert len(vectors) == 4
    assert len(lengths) == 4
    # Check if sorted by distance
    assert lengths[0] <= lengths[1] <= lengths[2] <= lengths[3]


def test_get_bond_angles():
    # Test with orthogonal vectors
    neighbor_vecs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),  # Zero vector for padding
    ]

    angles = get_bond_angles(neighbor_vecs)
    assert len(angles) == 6  # Number of unique pairs
    # Check orthogonal angles
    assert angles[0] == pytest.approx(np.pi / 2)  # Between first two vectors


def test_compute_descriptors():
    # Create a simple molecule
    mol = Chem.MolFromSmiles("CCO")
    AllChem.EmbedMolecule(mol, randomSeed=42)

    descriptors = compute_descriptors(mol)
    assert descriptors.shape[0] == mol.GetNumAtoms()
    # Each descriptor should have generation descriptors (4) +
    # neighbor distances (4) + bond angles (6)
    assert descriptors.shape[1] == 14


def test_edge_cases():
    # Test zero displacement
    point = np.array([0.0, 0.0, 0.0])
    origin = np.array([0.0, 0.0, 0.0])
    ref1 = np.array([1.0, 0.0, 0.0])
    ref2 = np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError):
        calculate_spherical_coords(point, origin, ref1, ref2)

    # Test collinear reference vectors
    point = np.array([1.0, 0.0, 0.0])
    ref2_collinear = np.array([2.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        calculate_spherical_coords(point, origin, ref1, ref2_collinear)

    # Test get_neighbor_vectors with single atom
    coords = np.array([[0.0, 0.0, 0.0]])
    vectors, lengths = get_neighbor_vectors(coords, 0)
    assert len(vectors) == 4
    assert all(np.array_equal(v, np.zeros(3)) for v in vectors[0:])
