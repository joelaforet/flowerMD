"""Prepare supplied all-atom structures for native DPD force evaluation."""

from collections.abc import Mapping
from copy import deepcopy

import mbuild as mb
import numpy as np
import unyt as u

from flowermd.internal.aa_snapshot import create_all_atom_frame
from flowermd.internal.bonded_assignment import own_bonded_assignment
from flowermd.internal.openff_gmso import _validate_graph
from flowermd.library.aa_dpd import AllAtomDPD
from flowermd.library.bonded_providers import resolve_bonded_provider
from flowermd.library.systems import mbuildSystem


class AllAtomSystem(mbuildSystem):
    """Prepare one supplied connected explicit-H structure without moving it.

    ``compound`` supplies mBuild particle order, origin-based nm coordinates
    and an orthorhombic box. ``molecule`` is the authoritative RDKit graph;
    ``atom_map`` maps its atom indices to the supplied particle order.
    The constructor copies all inputs before mBuild conversion and validates
    the mapped elements and bond graph. It preserves graph stereo labels;
    it does not certify intended stereochemistry or audit distorted geometry.

    Call apply_dpd before accessing prepared forces, frame or assignment report.
    References are fixed at 1 angstrom, 1 kcal/mol and 1 amu. Reapplication
    always starts from the retained construction input, including coordinates.
    Edits to exposed mBuild/GMSO objects do not redefine that input. Arbitrary
    edits to the exposed frame or force objects are unsupported. Existing
    Simulation instances keep their previous state and forces after this
    system is prepared again. No walk, DPD trajectory, FIRE minimization or
    initialization stopping rule runs here.
    """

    def __init__(self, compound, *, molecule, atom_map):
        from rdkit import Chem

        if not isinstance(compound, mb.Compound) or compound.n_particles == 0:
            raise ValueError("compound must be a nonempty mBuild Compound")
        if not isinstance(molecule, Chem.Mol) or molecule.GetNumAtoms() == 0:
            raise ValueError("molecule must be a nonempty RDKit molecule")
        graph = Chem.Mol(molecule)
        Chem.SanitizeMol(graph)
        if len(Chem.GetMolFrags(graph)) != 1:
            raise ValueError(
                "AllAtomSystem currently requires a connected graph"
            )
        if any(
            a.GetNumImplicitHs() or a.GetNumExplicitHs()
            for a in graph.GetAtoms()
        ):
            raise ValueError(
                "molecule must contain hydrogens as explicit atoms"
            )
        if not isinstance(atom_map, Mapping):
            raise ValueError("atom_map must be a complete integer bijection")
        mapping = deepcopy(dict(atom_map))
        if compound.box is None:
            raise ValueError("compound requires an orthorhombic box")
        lengths = np.array(compound.box.lengths, dtype=float, copy=True)
        if (
            lengths.shape != (3,)
            or not np.all(np.isfinite(lengths))
            or np.any(lengths <= 0)
            or not np.array_equal(compound.box.angles, [90, 90, 90])
        ):
            raise ValueError(
                "compound requires positive finite orthorhombic box lengths"
            )
        positions = np.array(compound.xyz, dtype=float, copy=True)
        if positions.shape != (compound.n_particles, 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError(
                "compound coordinates must be finite N-by-3 nm values"
            )
        owned = mb.clone(compound)
        particles = tuple(owned.particles())
        particle_indices = {p: i for i, p in enumerate(particles)}
        edges = {
            tuple(sorted((particle_indices[a], particle_indices[b])))
            for a, b in owned.bonds()
        }
        super().__init__(
            owned,
            base_units={
                "length": 1 * u.angstrom,
                "energy": 1 * u.kcal / u.mol,
                "mass": 1 * u.amu,
            },
        )
        sites = tuple(self.gmso_system.sites)
        site_indices = {s: i for i, s in enumerate(sites)}
        if (
            tuple(self.system.particles()) != particles
            or len(sites) != len(particles)
            or not np.array_equal(
                self.gmso_system.positions.to_value("nm"), positions
            )
            or any(
                s.element is None
                or p.element is None
                or s.element.atomic_number != p.element.atomic_number
                for s, p in zip(sites, particles)
            )
            or {
                tuple(sorted(site_indices[s] for s in b.connection_members))
                for b in self.gmso_system.bonds
            }
            != edges
        ):
            raise ValueError(
                "mBuild conversion changed particle order, coordinates or graph"
            )
        graph_edges = {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in graph.GetBonds()
        }
        _validate_graph(self.gmso_system, graph, mapping, graph_edges)
        self._construction_topology = deepcopy(self.gmso_system)
        self._construction_positions = positions
        self._construction_box = lengths
        self._molecule = graph
        self._atom_map = mapping
        self._dpd_forcefield = None
        self._assignment_report = None

    def apply_dpd(
        self,
        *,
        bonded="uff",
        force_field=None,
        repulsion,
        gamma,
        kT,
        r_cut,
        bonded_scale,
        epsilon_weighting=True,
        epsilon_reference=None,
        conservative=False,
        include_bonds=True,
        include_angles=True,
        include_torsions=True,
        include_impropers=True,
    ):
        """Assign a fresh topology, force bundle and matching GSD frame.

        bonded accepts a provider with a callable assign method, such as
        UFFProvider, OpenFFProvider or SageProvider. Providers assign native
        GMSO parameters and report their source. Each call receives fresh
        input copies; returned data is validated and copied before use.

        The strings uff, openff and sage remain supported. force_field only
        configures string inputs. OpenFF defaults to openff-2.3.0.offxml;
        sage fixes that resource. OpenFF vdW assignment follows
        epsilon_weighting. UFF also extracts nonbonded parameters when
        unweighted, but DPD does not consume them. Providers assign the terms
        in their documented method independently of the four ablations. UFFProvider
        omits inversions because no maintained execution backend is supported.
        include_impropers=True executes supported assigned groups; it does not
        require a provider to assign omitted forms. OpenFF periodic impropers
        remain supported. The report distinguishes requested flags, assigned
        and executed groups, and backend force-object counts.

        Coefficients, units, strict boolean ablations and weighting follow
        AllAtomDPD. Disabled terms preserve groups and pair exclusions.
        Assignment and frame construction finish before replacing any prepared
        state. A failed call leaves the preceding configuration usable.
        Successful reapplication creates fresh forces for a new Simulation.
        """
        from rdkit import Chem

        if not isinstance(epsilon_weighting, bool):
            raise ValueError("epsilon_weighting must be a bool")
        provider = resolve_bonded_provider(bonded, force_field)
        result = provider.assign(
            deepcopy(self._construction_topology),
            Chem.Mol(self._molecule),
            atom_map=deepcopy(self._atom_map),
            assign_nonbonded=epsilon_weighting,
        )
        typed, assignment = own_bonded_assignment(
            result, self._construction_topology, self._molecule, self._atom_map
        )
        options = dict(
            repulsion=repulsion,
            gamma=gamma,
            kT=kT,
            r_cut=r_cut,
            bonded_scale=bonded_scale,
            epsilon_weighting=epsilon_weighting,
            epsilon_reference=epsilon_reference,
            conservative=conservative,
            include_bonds=include_bonds,
            include_angles=include_angles,
            include_torsions=include_torsions,
            include_impropers=include_impropers,
        )
        bundle = AllAtomDPD(typed, **options)
        frame = create_all_atom_frame(
            typed,
            type_labels=bundle.type_labels,
            positions_nm=self._construction_positions,
            box_lengths_nm=self._construction_box,
        )
        references = deepcopy(bundle.reference_values)
        snapshot_references = deepcopy(references)
        force_references = deepcopy(references)
        reference = (
            (
                max(
                    float(p.parameters["epsilon"].to_value("kcal/mol"))
                    for p in bundle.type_labels["sites"]
                )
                if epsilon_reference is None
                else float(epsilon_reference)
            )
            if epsilon_weighting
            else None
        )
        report = {
            "bonded": bonded
            if isinstance(bonded, str)
            else f"{type(provider).__module__}.{type(provider).__qualname__}",
            "assignment": assignment,
            "dpd": deepcopy(options),
            "execution": deepcopy(bundle.execution_summary),
            "epsilon_source": assignment["source"]
            if epsilon_weighting
            else None,
            "epsilon_reference_kcal_mol": reference,
            "pair_coefficients": {
                key: dict(value)
                for key, value in bundle.forces_by_category["pair"][
                    0
                ].params.items()
            },
            "reference_values": deepcopy(references),
        }
        self.gmso_system = typed
        self._dpd_forcefield = bundle
        self._hoomd_snapshot = frame
        self._hoomd_forcefield = list(bundle.hoomd_forces)
        self._reference_values = references
        self._snap_refs = snapshot_references
        self._ff_refs = force_references
        self._assignment_report = report

    def _require_prepared(self):
        if self._dpd_forcefield is None:
            raise RuntimeError(
                "call apply_dpd before accessing the prepared system"
            )

    @property
    def hoomd_snapshot(self):
        """Return the prepared native GSD frame."""
        self._require_prepared()
        return self._hoomd_snapshot

    @property
    def hoomd_forcefield(self):
        """Return a fresh list of the prepared HOOMD force objects."""
        self._require_prepared()
        return list(self._hoomd_forcefield)

    @property
    def dpd_forcefield(self):
        """Return the prepared AllAtomDPD bundle and semantic force categories."""
        self._require_prepared()
        return self._dpd_forcefield

    @property
    def assignment_report(self):
        """Return a copy of assignment provenance and effective DPD coefficients."""
        self._require_prepared()
        return deepcopy(self._assignment_report)

    @property
    def reference_values(self):
        """Return copies of the fixed angstrom, kcal/mol and amu references."""
        return deepcopy(self._reference_values)

    @reference_values.setter
    def reference_values(self, value):
        raise ValueError("AllAtomSystem references are fixed")

    @property
    def reference_length(self):
        """Return the fixed 1 angstrom length reference."""
        return self.reference_values["length"]

    @reference_length.setter
    def reference_length(self, value):
        raise ValueError("AllAtomSystem references are fixed")

    @property
    def reference_energy(self):
        """Return the fixed 1 kcal/mol energy reference."""
        return self.reference_values["energy"]

    @reference_energy.setter
    def reference_energy(self, value):
        raise ValueError("AllAtomSystem references are fixed")

    @property
    def reference_mass(self):
        """Return the fixed 1 amu mass reference."""
        return self.reference_values["mass"]

    @reference_mass.setter
    def reference_mass(self, value):
        raise ValueError("AllAtomSystem references are fixed")

    @property
    def auto_scale(self):
        """Return False because automatic reference scaling is unsupported."""
        return False

    @auto_scale.setter
    def auto_scale(self, value):
        if value is not False:
            raise ValueError("AllAtomSystem does not support automatic scaling")

    def apply_forcefield(self, *args, **kwargs):
        """Reject generic force-field assignment; use apply_dpd."""
        raise ValueError("AllAtomSystem requires apply_dpd")

    def remove_hydrogens(self):
        """Reject hydrogen removal because it invalidates the chemical mapping."""
        raise ValueError("AllAtomSystem requires explicit hydrogens")
