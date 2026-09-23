"""SMIRNOFF (Sage) bonded and pair parameters for the all-atom DPD force field.

The route is OpenFF Toolkit -> Interchange -> OpenMM System -> tables. GMSO
has no importer for SMIRNOFF parameters, and going through ParmEd keeps only
the first Fourier term of each torsion and loses the improper atom order, so
the OpenMM System is read directly. No Coulomb or Lennard-Jones terms are
exported; per-atom LJ epsilons are kept only to weight the DPD coefficients.

Partial charges are set to zero before Interchange is built, which skips the
AM1-BCC step that DPD does not need. Use `openff_topology` on the resulting
force field to build a fully charged Interchange for a downstream hand-off.

Requires rdkit, openff-toolkit and openff-interchange (imported lazily).
"""

import numpy as np

from flowermd.internal.all_atom_parameters import (
    AllAtomParameters,
    compound_box_lengths_a,
    compound_to_rdkit,
    intern_type,
    molecular_compounds,
    strip_keys,
)

KJ_TO_KCAL = 1.0 / 4.184


def compound_to_openff_topology(compound):
    """Return an OpenFF `Topology` for `compound` with zero partial charges.

    One `Molecule` per disconnected chain, coordinates carried over, and
    stereochemistry read from the 3D coordinates.
    """
    from openff.toolkit import Molecule, Topology
    from openff.units import unit

    molecules = []
    for chain in molecular_compounds(compound):
        molecule = Molecule.from_rdkit(
            compound_to_rdkit(chain),
            allow_undefined_stereo=True,
            hydrogens_are_explicit=True,
        )
        molecule._conformers = []
        molecule.add_conformer(
            np.asarray(chain.xyz, dtype=float) * unit.nanometer
        )
        molecule.partial_charges = (
            np.zeros(molecule.n_atoms) * unit.elementary_charge
        )
        molecules.append(molecule)
    return Topology.from_molecules(molecules)


def _torsion_kind(group, bond_edges):
    """Classify a four-atom torsion as proper, improper or unclassified."""
    path = all(
        tuple(sorted(pair)) in bond_edges for pair in zip(group, group[1:])
    )
    star = any(
        all(
            tuple(sorted((center, other))) in bond_edges
            for other in group
            if other != center
        )
        for center in group
    )
    if path and not star:
        return "proper"
    if star and not path:
        return "improper"
    return "unclassified"


def parameterize_openff(compound, force_field="openff-2.3.0.offxml"):
    """Return `AllAtomParameters` for `compound` from a SMIRNOFF force field.

    Parameters
    ----------
    compound : mbuild.Compound, required
        All-atom compound with elements on every particle and a periodic
        ``box``. Coordinates in nm.
    force_field : str, default "openff-2.3.0.offxml"
        SMIRNOFF force field name or path (Sage 2.3.0 by default).

    Returns
    -------
    parameters : AllAtomParameters
    topology : openff.toolkit.Topology
        The topology the parameters were assigned to.

    """
    import openmm
    from openff.interchange import Interchange
    from openff.toolkit import ForceField
    from openff.units import unit as offunit
    from openmm import unit

    box_lengths_a = compound_box_lengths_a(compound)
    topology = compound_to_openff_topology(compound)
    interchange = Interchange.from_smirnoff(
        ForceField(force_field),
        topology,
        # One representative per distinct molecule, as Interchange requires.
        charge_from_molecules=list(topology.unique_molecules),
        box=np.diag(box_lengths_a / 10.0) * offunit.nanometer,
        positions=np.asarray(compound.xyz, dtype=float) * offunit.nanometer,
    )
    # Constrained hydrogens must become flexible harmonic bonds for DPD.
    system = interchange.to_openmm_system(add_constrained_forces=True)

    n_atoms = system.getNumParticles()
    masses = np.asarray(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(n_atoms)
        ]
    )
    result = AllAtomParameters(
        positions_a=np.asarray(compound.xyz, dtype=float) * 10.0,
        box_lengths_a=box_lengths_a,
        masses_amu=masses,
        source="openff",
    )

    bond_table, angle_table, torsion_table = {}, {}, {}
    bond_edges = set()
    torsions = []
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for idx in range(force.getNumBonds()):
                i, j, r0, k = force.getBondParameters(idx)
                group = (int(i), int(j))
                bond_edges.add(tuple(sorted(group)))
                # OpenMM: E = k/2 (r - r0)^2, same as HOOMD Harmonic.
                k = float(
                    k.value_in_unit(
                        unit.kilocalorie_per_mole / unit.angstrom**2
                    )
                )
                r0 = float(r0.value_in_unit(unit.angstrom))
                name = intern_type(
                    bond_table, "openff_bond_", (k, r0), {"k": k, "r0": r0}
                )
                result.bonds.append(group)
                result.bond_types.append(name)
        elif isinstance(force, openmm.HarmonicAngleForce):
            for idx in range(force.getNumAngles()):
                i, j, k_, t0, k = force.getAngleParameters(idx)
                k = float(
                    k.value_in_unit(unit.kilocalorie_per_mole / unit.radian**2)
                )
                t0 = float(t0.value_in_unit(unit.radian))
                name = intern_type(
                    angle_table, "openff_angle_", (k, t0), {"k": k, "t0": t0}
                )
                result.angles.append((int(i), int(j), int(k_)))
                result.angle_types.append(name)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for idx in range(force.getNumTorsions()):
                torsions.append(force.getTorsionParameters(idx))
        elif isinstance(force, openmm.NonbondedForce):
            nonbonded = force
        elif isinstance(force, openmm.CMMotionRemover):
            continue
        else:
            raise ValueError(
                f"Unsupported OpenMM force in the SMIRNOFF export: "
                f"{type(force).__name__}."
            )
    if nonbonded is None:
        raise ValueError("The SMIRNOFF export has no NonbondedForce.")

    for i, j, k_, ell, periodicity, phase, strength in torsions:
        group = (int(i), int(j), int(k_), int(ell))
        kind = _torsion_kind(group, bond_edges)
        # OpenMM: E = k (1 + cos(n phi - phi0)); HOOMD Periodic:
        # E = k/2 (1 + d cos(n phi - phi0)), so k_hoomd = 2 k, d = 1.
        k = 2.0 * float(strength.value_in_unit(unit.kilocalorie_per_mole))
        phi0 = float(phase.value_in_unit(unit.radian)) % (2.0 * np.pi)
        values = {"k": k, "n": int(periodicity), "d": 1, "phi0": phi0}
        if kind == "improper":
            name = intern_type(
                torsion_table,
                "openff_improper_",
                (k, int(periodicity), phi0, kind),
                values,
            )
            result.impropers.append(group)
            result.improper_types.append(name)
        else:
            name = intern_type(
                torsion_table,
                f"openff_{kind}_",
                (k, int(periodicity), phi0, kind),
                values,
            )
            result.dihedrals.append(group)
            result.dihedral_types.append(name)
    result.bond_params = strip_keys(bond_table)
    result.angle_params = strip_keys(angle_table)
    all_torsions = strip_keys(torsion_table)
    dihedral_names = set(result.dihedral_types)
    improper_names = set(result.improper_types)
    result.dihedral_params = {
        n: v for n, v in all_torsions.items() if n in dihedral_names
    }
    result.improper_params = {
        n: v for n, v in all_torsions.items() if n in improper_names
    }

    particle_table = {}
    for i in range(n_atoms):
        _, sigma, epsilon = nonbonded.getParticleParameters(i)
        epsilon = float(epsilon.value_in_unit(unit.kilocalorie_per_mole))
        sigma = float(sigma.value_in_unit(unit.angstrom))
        name = intern_type(
            particle_table,
            "openff_",
            (sigma, epsilon, float(masses[i])),
            {"epsilon": epsilon},
        )
        result.particle_types.append(name)
    result.particle_epsilons = {
        name: entry["epsilon"] for name, entry in particle_table.items()
    }
    result.particle_keys = {
        name: entry["_key"] for name, entry in particle_table.items()
    }
    result.epsilon_ref = max(result.particle_epsilons.values())
    return result, topology
