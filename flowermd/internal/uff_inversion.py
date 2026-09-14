"""Exact Wilson inversion forces for single-rank CPU HOOMD simulations."""

from numbers import Integral

import hoomd
import numpy as np
import sympy

_WILSON_EXPRESSION = sympy.sympify("k*(c0+c1*cos(omega)+c2*cos(2*omega))")


def is_uff_inversion(potential):
    """Identify the assigned Wilson form and its ordered member convention."""
    return (
        potential is not None
        and potential.tags.get("form") == "uff_inversion"
        and potential.tags.get("coordinate") == "wilson_out_of_plane"
        and potential.tags.get("member_convention")
        == "center,plane1,plane2,out"
        and potential.independent_variables == {sympy.Symbol("omega")}
        and sympy.simplify(potential.expression - _WILSON_EXPRESSION) == 0
    )


def wilson_energy_forces(vectors, coefficients):
    """Return one ordered term's energy and four Cartesian forces.

    The three vectors point from the center to plane1, plane2 and the out
    atom in angstrom. Coefficients are scaled k in kcal/mol and c0, c1, c2.
    An exactly zero k skips geometry evaluation. Singular active coordinates
    and nonfinite results raise ValueError. No denominator floors or clipping
    alter the Wilson coordinate.
    """
    k, c0, c1, c2 = coefficients
    if k == 0:
        return 0.0, np.zeros((4, 3))
    a, b, c = vectors
    normal = np.cross(a, b)
    normal_length = np.linalg.norm(normal)
    out_length = np.linalg.norm(c)
    if (
        not np.isfinite(normal_length)
        or not np.isfinite(out_length)
        or normal_length == 0
        or out_length == 0
    ):
        raise ValueError(
            "undefined UFF out-of-plane bending: atoms coincide or the "
            "plane atoms are collinear"
        )
    h, t = normal / normal_length, c / out_length
    s = np.dot(h, t)
    # This is sqrt(1-s*s) for unit vectors, without cancellation near |s|=1.
    q = np.linalg.norm(np.cross(h, t))
    if c1 != 0 and q == 0:
        raise ValueError(
            "nondifferentiable UFF out-of-plane bending: the out atom is "
            "exactly perpendicular to the plane, so there is no unique force"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        energy = k * (c0 + c1 * q + c2 * (1 - 2 * s * s))
        derivative = k * (-4 * c2 * s - (c1 * s / q if c1 != 0 else 0))
        normal_gradient = (t - s * h) / normal_length
        gradients = np.array(
            [
                np.cross(b, normal_gradient),
                np.cross(normal_gradient, a),
                (h - s * t) / out_length,
            ]
        )
        outer_forces = -derivative * gradients
        forces = np.vstack((-outer_forces.sum(axis=0), outer_forces))
    if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
        raise ValueError(
            "UFF inversion energy or forces exceed finite float range"
        )
    return float(energy), forces


class UFFInversionForce(hoomd.md.force.Custom):
    """Evaluate copied center-first Wilson terms on single-rank CPU states.

    Groups contain particle tags; each callback resolves the current local
    indices through rtag. Coefficients contain k, c0, c1 and c2 for each group.
    k already includes the assignment's division among the three ordered
    terms and the caller's bonded scale. Energy and virial are distributed
    equally among the four atoms. Forces scatter-add when groups share atoms.

    Only orthorhombic 3D boxes on one CPU rank are supported. Minimum-image
    center-relative vectors define the geometry. GPU and MPI execution require
    separate backends. Changing particle count after attachment is unsupported.
    """

    def __init__(self, groups, coefficients, n_particles):
        super().__init__()
        raw_groups = np.asarray(groups)
        if (
            not isinstance(n_particles, Integral)
            or isinstance(n_particles, bool)
            or n_particles <= 0
            or raw_groups.ndim != 2
            or raw_groups.shape[1] != 4
            or raw_groups.dtype.kind not in "iu"
        ):
            raise ValueError(
                "UFF inversion groups must be integer N-by-4 particle tags"
            )
        self._groups = np.array(groups, dtype=np.int64, copy=True).reshape(
            -1, 4
        )
        self._coefficients = np.array(
            coefficients, dtype=float, copy=True
        ).reshape(-1, 4)
        self._n_particles = n_particles
        if (
            len(self._groups) != len(self._coefficients)
            or np.any(self._groups < 0)
            or np.any(self._groups >= n_particles)
            or any(len(set(group)) != 4 for group in self._groups)
        ):
            raise ValueError(
                "UFF inversion groups require four distinct valid particle tags"
            )
        if not np.all(np.isfinite(self._coefficients)) or np.any(
            self._coefficients[:, 0] < 0
        ):
            raise ValueError(
                "UFF inversion coefficients must be finite with nonnegative k"
            )

    def _check_box(self):
        box = self._simulation.state.box
        lengths = np.array(box.L, dtype=float)
        if (
            box.dimensions != 3
            or any(value != 0 for value in (box.xy, box.xz, box.yz))
            or not np.all(np.isfinite(lengths))
            or np.any(lengths <= 0)
        ):
            raise ValueError(
                "UFF out-of-plane bending requires a finite positive "
                "orthorhombic 3D box with right angles. Use that box or set "
                "include_impropers=False."
            )
        return lengths

    def _attach_hook(self):
        if not isinstance(self._simulation.device, hoomd.device.CPU):
            raise RuntimeError(
                "UFF out-of-plane bending supports CPU devices only. "
                "Use hoomd.device.CPU() or set include_impropers=False."
            )
        if self._simulation.device.communicator.num_ranks != 1:
            raise RuntimeError(
                "UFF out-of-plane bending supports a single MPI rank only. "
                "Use one rank or set include_impropers=False."
            )
        self._check_box()
        if self._simulation.state.N_particles != self._n_particles:
            raise ValueError(
                "UFF inversion particle count differs from its assigned topology"
            )
        super()._attach_hook()

    def set_forces(self, timestep):
        """Compute fresh force, energy and virial arrays for the current order."""
        lengths = self._check_box()
        with self._state.cpu_local_snapshot as snapshot:
            positions = np.asarray(snapshot.particles.position)
            reverse_tags = np.asarray(snapshot.particles.rtag)
            if (
                len(positions) != self._n_particles
                or len(reverse_tags) != self._n_particles
            ):
                raise ValueError(
                    "UFF inversion particle count changed after attachment"
                )
            force = np.zeros((self._n_particles, 3))
            energy = np.zeros(self._n_particles)
            virial = np.zeros((self._n_particles, 6))
            for tags, coefficients in zip(self._groups, self._coefficients):
                if coefficients[0] == 0:
                    continue
                group = reverse_tags[tags]
                if np.any(group >= self._n_particles):
                    raise ValueError(
                        "UFF inversion particle tags are not local"
                    )
                vectors = positions[group[1:]] - positions[group[0]]
                vectors -= lengths * np.rint(vectors / lengths)
                term_energy, term_force = wilson_energy_forces(
                    vectors, coefficients
                )
                tensor = vectors.T @ term_force[1:]
                term_virial = tensor[(0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
                np.add.at(force, group, term_force)
                np.add.at(energy, group, term_energy / 4)
                np.add.at(virial, group, term_virial / 4)
            if not all(
                np.all(np.isfinite(value)) for value in (force, energy, virial)
            ):
                raise ValueError(
                    "accumulated UFF inversion values exceed finite float range"
                )
        with self.cpu_local_force_arrays as arrays:
            arrays.force[:] = force
            arrays.potential_energy[:] = energy
            arrays.virial[:] = virial
            arrays.torque[:] = 0
