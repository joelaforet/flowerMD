"""Geometry helpers for placing all-atom chains in a box without packing.

Both all-atom placement systems treat a chain's repeat units (the children of
the chain compound built by `flowermd.base.Polymer`) as rigid bodies. They
move and rotate whole repeats, never individual atoms, so every bond, angle
and torsion inside a repeat keeps the geometry mBuild built. Only the bonds
between consecutive repeats are stretched or compressed, and the DPD/FIRE
stage relaxes those.

Coordinates are left unwrapped; HOOMD wraps particles into the box.
"""

import math

import numpy as np
import unyt as u

from flowermd.utils import (
    get_target_box_mass_density,
    get_target_box_number_density,
)

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


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


def rotation_between(source, target):
    """Proper rotation taking unit(`source`) onto unit(`target`).

    The rotation is about the axis perpendicular to both vectors, so it adds
    no arbitrary spin about the target direction.
    """
    source = np.asarray(source, dtype=float) / np.linalg.norm(source)
    target = np.asarray(target, dtype=float) / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        # Antiparallel: rotate by pi about any axis perpendicular to source.
        helper = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


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


def junction_atoms(chain, repeats):
    """For each repeat, the atoms bonded to the previous and next repeat.

    Returns a list of ``(inbound_atom, outbound_atom)`` pairs; an entry is
    None where the repeat has no neighbour on that side (chain ends).
    """
    owner = {}
    for index, repeat in enumerate(repeats):
        for particle in repeat.particles():
            owner[particle] = index
    inbound = [None] * len(repeats)
    outbound = [None] * len(repeats)
    for a, b in chain.bonds():
        ia, ib = owner[a], owner[b]
        if ia == ib:
            continue
        if ia > ib:
            a, b, ia, ib = b, a, ib, ia
        outbound[ia] = a
        inbound[ib] = b
    return list(zip(inbound, outbound))


def repeat_axis(repeat, inbound, outbound):
    """Unit vector along a repeat, from its inbound side to its outbound side.

    Falls back to the repeat's longest principal axis when it has fewer than
    two junction atoms (chain ends and single-repeat chains), pointed away
    from the one junction atom it does have.
    """
    xyz = np.asarray(repeat.xyz, dtype=float)
    center = xyz.mean(axis=0)
    if inbound is not None and outbound is not None:
        axis = np.asarray(outbound.pos) - np.asarray(inbound.pos)
        if np.linalg.norm(axis) > 1e-9:
            return axis / np.linalg.norm(axis)
    if len(xyz) < 2:
        return np.array([0.0, 0.0, 1.0])
    _, _, vt = np.linalg.svd(xyz - center, full_matrices=False)
    axis = vt[0]
    if (
        inbound is not None
        and np.dot(axis, center - np.asarray(inbound.pos)) < 0
    ):
        axis = -axis
    elif (
        outbound is not None
        and np.dot(axis, np.asarray(outbound.pos) - center) < 0
    ):
        axis = -axis
    return axis


def repeat_step_length(chain, repeats, junctions):
    """Mean centroid-to-centroid distance between consecutive repeats (nm).

    This is the step the random walk takes so that junction bonds start
    near their built length. Chains with one repeat return 0.
    """
    if len(repeats) < 2:
        return 0.0
    centroids = [np.asarray(r.xyz, dtype=float).mean(axis=0) for r in repeats]
    steps = [
        np.linalg.norm(centroids[i + 1] - centroids[i])
        for i in range(len(centroids) - 1)
    ]
    return float(np.mean(steps))


def random_walk_path(n_steps, step, start, rng, max_turn):
    """Positions of an isotropic random walk with a bounded turning angle.

    Parameters
    ----------
    n_steps : int
        Number of positions to return.
    step : float
        Step length.
    start : array-like, shape (3,)
        First position.
    rng : numpy.random.Generator
    max_turn : float
        Largest allowed angle (radians) between consecutive steps. Pi allows
        immediate back-folding; pi/2 keeps the walk moving forward.

    """
    positions = np.empty((n_steps, 3))
    positions[0] = start
    previous = None
    for i in range(1, n_steps):
        while True:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            if previous is None:
                break
            cosine = float(np.dot(direction, previous))
            if math.acos(max(-1.0, min(1.0, cosine))) <= max_turn:
                break
        positions[i] = positions[i - 1] + step * direction
        previous = direction
    return positions


def place_along_path(repeats, junctions, path, twist=GOLDEN_ANGLE):
    """Move each repeat rigidly so its axis follows the local path tangent.

    Repeat ``i`` is centered on ``path[i]``, its inbound-to-outbound axis is
    rotated onto the tangent through ``path[i]``, and an additional twist of
    ``i * twist`` about the tangent spreads substituents around the chain.
    """
    n = len(repeats)
    for i, (repeat, (inbound, outbound)) in enumerate(zip(repeats, junctions)):
        if n == 1:
            tangent = None
        elif i == 0:
            tangent = path[1] - path[0]
        elif i == n - 1:
            tangent = path[-1] - path[-2]
        else:
            tangent = path[i + 1] - path[i - 1]
        xyz = np.asarray(repeat.xyz, dtype=float)
        center = xyz.mean(axis=0)
        if tangent is None or np.linalg.norm(tangent) < 1e-12:
            repeat.xyz = xyz - center + path[i]
            continue
        rotation = rotation_between(
            repeat_axis(repeat, inbound, outbound), tangent
        )
        rotation = axial_rotation(tangent, i * twist) @ rotation
        repeat.xyz = (xyz - center) @ rotation.T + path[i]


def serpentine_lattice_sites(count, box_lengths):
    """Cell-centre sites of a near-cubic grid, ordered along a serpentine path.

    Consecutive sites are adjacent, so consecutive repeats of a chain land
    next to each other. Returns ``(sites, grid_shape)``; unused sites are at
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
