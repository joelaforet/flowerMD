"""Geometry helpers for placing all-atom chains in a box without packing.

Both all-atom placement systems keep the chemistry mBuild built: every bond
length, bond angle and tetrahedral center (its handedness) of the input is
preserved. The random walk only rotates about bonds between repeat units,
which changes the torsions about those bonds and nothing else; the lattice
moves whole chains as rigid bodies.

Coordinates are left unwrapped; HOOMD wraps particles into the box.
"""

import math

import numpy as np
import unyt as u

from flowermd.utils import (
    get_target_box_mass_density,
    get_target_box_number_density,
)


def target_box_lengths(density, mass, n_particles):
    """Cubic box lengths (nm, as a numpy array) for a mass or number density.

    Parameters
    ----------
    density : unyt.unyt_quantity, required
        Mass density (e.g. g/cm**3) or number density (e.g. nm**-3).
    mass : unyt.unyt_quantity, required
        Total mass of the system, used for mass densities.
    n_particles : int, required
        Particle count, used for number densities.

    """
    mass_dims = (u.Unit("kg") / u.Unit("m**3")).dimensions
    number_dims = u.Unit("m**-3").dimensions
    if density.units.dimensions == mass_dims:
        box = get_target_box_mass_density(density=density, mass=mass)
    elif density.units.dimensions == number_dims:
        box = get_target_box_number_density(
            density=density, n_beads=n_particles
        )
    else:
        raise ValueError(
            f"Density dimensions of {density.units.dimensions} were given, "
            "but only mass density and number density are supported."
        )
    return np.asarray(box.to("nm").value, dtype=float)


def random_rotation(rng):
    """Uniformly random proper rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def axial_rotation(axis, angle):
    """Rotation matrix for `angle` radians about the unit vector `axis`."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * skew
    )


def repeat_units(chain):
    """Return the chain's repeat compounds in backbone order.

    Uses the chain's children (mBuild's `Polymer` recipe adds one child per
    repeat, in sequence). A chain with no children is one rigid unit.
    """
    children = [child for child in chain.children if child.n_particles]
    return children if children else [chain]


def _far_side(adjacency, near, far):
    """Atoms reached from `far` without crossing the bond (`near`, `far`).

    Returns None when `near` is reached as well, i.e. the bond is part of a
    ring and no rigid rotation about it exists.
    """
    seen = {far}
    stack = [far]
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if i == far and j == near:
                continue
            if j == near:
                return None
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return np.fromiter(seen, dtype=int)


def random_walk_conformation(xyz, units, bonds, start, rng):
    """A random conformation of one molecule by rotations about junctions.

    The molecule is first turned by a uniformly random rotation and moved so
    the centroid of its first unit is at `start`. Then, for every bond between
    two units, the part of the molecule on the far side of the bond is
    rotated about the bond by an angle drawn uniformly from [0, 2*pi). A
    rotation about a bond changes only the torsions about that bond, so every
    bond length, bond angle and tetrahedral center keeps its input geometry,
    including the handedness of stereocenters next to a junction. A junction
    that is part of a ring (the two-bond junctions of a ladder polymer) has no
    free torsion and keeps its input geometry.

    Parameters
    ----------
    xyz : (N, 3) array
        Input coordinates of the molecule.
    units : list of lists of int
        Atom indices of each unit (repeat), in backbone order.
    bonds : (n_bonds, 2) array of int
        Bonds of the molecule as atom-index pairs.
    start : (3,) array
        Position of the first unit's centroid.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
        New (N, 3) coordinates.

    """
    xyz = np.asarray(xyz, dtype=float)
    owner = np.full(len(xyz), -1, dtype=int)
    for unit_id, members in enumerate(units):
        owner[members] = unit_id
    adjacency = [[] for _ in range(len(xyz))]
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    center = xyz[units[0]].mean(axis=0)
    xyz = (xyz - center) @ random_rotation(rng).T + start
    for a, b in bonds:
        if owner[a] == owner[b]:
            continue
        if owner[a] > owner[b]:
            a, b = b, a
        side = _far_side(adjacency, a, b)
        if side is None:
            continue
        rotation = axial_rotation(
            xyz[b] - xyz[a], rng.uniform(0.0, 2 * math.pi)
        )
        xyz[side] = (xyz[side] - xyz[a]) @ rotation.T + xyz[a]
    return xyz


def serpentine_lattice_sites(count, box_lengths):
    """Cell-center sites of a near-cubic grid, ordered along a serpentine path.

    Consecutive sites are adjacent. Returns ``(sites, grid_shape)``; unused sites are at
    the end of the traversal.
    """
    if count < 1:
        raise ValueError("count must be a positive integer.")
    box = np.asarray(box_lengths, dtype=float)
    n = math.ceil(count ** (1.0 / 3.0))
    while n**3 < count:
        n += 1
    dims = np.array([n, n, math.ceil(count / n**2)], dtype=int)
    plane = [
        (x, y)
        for y in range(dims[1])
        for x in (range(dims[0]) if y % 2 == 0 else reversed(range(dims[0])))
    ]
    indices = [
        (x, y, z)
        for z in range(dims[2])
        for x, y in (plane if z % 2 == 0 else reversed(plane))
    ]
    sites = (np.asarray(indices[:count], dtype=float) + 0.5) * box / dims
    return sites, dims
