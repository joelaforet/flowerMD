"""Check supplied-structure preparation through the public Simulation API."""

from itertools import product

import gsd.hoomd
import hoomd
import mbuild as mb
import numpy as np
import pytest
import unyt as u

from flowermd import Simulation
from flowermd.library import AllAtomSystem, mbuildSystem

Chem = pytest.importorskip("rdkit.Chem")


def inputs(smiles="CC(=O)NC"):
    compound = mb.load(smiles, smiles=True)
    compound.name = "original"
    compound.box = mb.Box(lengths=[3, 4, 5])
    compound.translate([1.5, 2, 2.5])
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return compound, molecule, dict(enumerate(range(molecule.GetNumAtoms())))


def create(smiles="CC(=O)NC"):
    compound, molecule, mapping = inputs(smiles)
    return AllAtomSystem(compound, molecule=molecule, atom_map=mapping)


def apply(system, **kwargs):
    options = dict(
        bonded="openff", repulsion=40, gamma=20, kT=1, r_cut=3, bonded_scale=0.7
    )
    options.update(kwargs)
    return system.apply_dpd(**options)


def test_constructor_owns_inputs_and_nonidentity_mapping():
    compound, molecule, mapping = inputs("F[C@H](Cl)C")
    outer = mb.Compound(name="parent")
    outer.add(compound)
    particles = tuple(compound.particles())
    positions = compound.xyz.copy()
    lengths = np.array(compound.box.lengths)
    bonds = tuple(compound.bonds())
    permutation = list(reversed(range(molecule.GetNumAtoms())))
    molecule = Chem.RenumberAtoms(molecule, permutation)
    mapping = dict(enumerate(permutation))
    molecule.SetProp("marker", "retained")
    before = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    system = AllAtomSystem(compound, molecule=molecule, atom_map=mapping)
    assert isinstance(system, mbuildSystem)
    assert compound.parent is outer and compound.name == "original"
    assert (
        tuple(compound.particles()) == particles
        and tuple(compound.bonds()) == bonds
    )
    np.testing.assert_array_equal(compound.xyz, positions)
    np.testing.assert_array_equal(compound.box.lengths, lengths)
    assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert mapping == dict(enumerate(permutation))
    assert set(system.system.particles()).isdisjoint(particles)
    assert system.box is not compound.box
    np.testing.assert_array_equal(
        system.gmso_system.positions.to_value("nm"), positions
    )
    assert [s.element.atomic_number for s in system.gmso_system.sites] == [
        p.element.atomic_number for p in particles
    ]
    # Later caller mutations cannot change the retained preparation input.
    molecule.RemoveAllConformers()
    molecule.GetAtomWithIdx(0).SetAtomicNum(0)
    mapping.clear()
    compound.translate([2, 2, 2])
    compound.box = mb.Box(lengths=[9, 9, 9])
    apply(system)
    np.testing.assert_array_equal(
        system.gmso_system.positions.to_value("nm"), positions
    )
    np.testing.assert_array_equal(
        system.hoomd_snapshot.configuration.box[:3], lengths * 10
    )


@pytest.mark.parametrize(
    "property_name",
    [
        "hoomd_snapshot",
        "hoomd_forcefield",
        "dpd_forcefield",
        "assignment_report",
    ],
)
def test_preparation_required(property_name):
    system = create("CC")
    with pytest.raises(RuntimeError, match="apply_dpd"):
        getattr(system, property_name)


@pytest.mark.parametrize(
    "change",
    [
        "map",
        "boolean_map",
        "element",
        "bond",
        "disconnected",
        "implicit_h",
        "box",
        "tilt",
        "coordinates",
    ],
)
def test_constructor_rejections_preserve_input(change):
    compound, molecule, mapping = inputs("CC")
    if change == "map":
        mapping.pop(0)
    elif change == "boolean_map":
        mapping = {True if i == 1 else i: j for i, j in mapping.items()}
    elif change == "element":
        molecule.GetAtomWithIdx(0).SetAtomicNum(14)
    elif change == "bond":
        compound.remove_bond(next(iter(compound.bonds())))
    elif change == "disconnected":
        molecule = Chem.AddHs(Chem.MolFromSmiles("C.C"))
    elif change == "implicit_h":
        molecule = Chem.MolFromSmiles("CC")
    elif change == "box":
        compound.box = None
    elif change == "tilt":
        compound.box = mb.Box(lengths=[3, 4, 5], angles=[90, 90, 85])
    else:
        xyz = compound.xyz.copy()
        xyz[0, 0] = float("nan")
        compound.xyz = xyz
    before_name = compound.name
    before_parent = compound.parent
    with pytest.raises(ValueError):
        AllAtomSystem(compound, molecule=molecule, atom_map=mapping)
    assert compound.name == before_name and compound.parent is before_parent


@pytest.mark.parametrize(
    "bonded,weighting", product(["uff", "openff", "sage"], [False, True])
)
def test_provider_weighting_and_provenance(bonded, weighting):
    system = create()
    result = apply(
        system,
        bonded=bonded,
        epsilon_weighting=weighting,
        include_impropers=bonded != "uff",
    )
    assert result is None
    report = system.assignment_report
    assert report["bonded"] == bonded
    expected_source = "UFF" if bonded == "uff" else "OpenFF SMIRNOFF"
    assert report["assignment"]["source"] == expected_source
    assert report["epsilon_source"] == (expected_source if weighting else None)
    assert report["dpd"]["epsilon_weighting"] is weighting
    if bonded != "uff":
        assert report["assignment"]["force_field"] == "openff-2.3.0.offxml"
        assert report["assignment"]["assign_nonbonded"] is weighting
        assert all(
            bool(s.atom_type.parameters) is weighting
            for s in system.gmso_system.sites
        )
    else:
        assert report["assignment"]["include_impropers"] is True
        assert system.gmso_system.n_impropers > 0
        assert system.dpd_forcefield.forces_by_category["impropers"] == ()
    bundle = system.dpd_forcefield
    pair = bundle.forces_by_category["pair"][0]
    assert report["pair_coefficients"] == {
        key: dict(value) for key, value in pair.params.items()
    }
    if weighting:
        eps = {
            label: float(p.parameters["epsilon"].to_value("kcal/mol"))
            for p, label in bundle.type_labels["sites"].items()
        }
        reference = max(eps.values())
        assert report["epsilon_reference_kcal_mol"] == reference
        for (a, b), coefficients in report["pair_coefficients"].items():
            factor = np.sqrt(eps[a] * eps[b]) / reference
            assert coefficients == pytest.approx(
                {"A": 40 * factor, "gamma": 20 * factor}
            )
    else:
        assert report["epsilon_reference_kcal_mol"] is None
        assert all(
            v == {"A": 40, "gamma": 20}
            for v in report["pair_coefficients"].values()
        )
    report["dpd"].clear()
    assert system.assignment_report["dpd"]
    forces = system.hoomd_forcefield
    forces.clear()
    assert system.hoomd_forcefield


@pytest.mark.parametrize("flags", product([True, False], repeat=4))
def test_all_independent_ablations(flags):
    system = create()
    names = (
        "include_bonds",
        "include_angles",
        "include_torsions",
        "include_impropers",
    )
    apply(system, epsilon_weighting=False, **dict(zip(names, flags)))
    frame = system.hoomd_snapshot
    assert (
        frame.bonds.N,
        frame.angles.N,
        frame.dihedrals.N,
        frame.impropers.N,
    ) == (11, 18, 22, 0)
    for kind, enabled in zip(
        ("bonds", "angles", "dihedrals", "impropers"), flags
    ):
        assert bool(system.dpd_forcefield.forces_by_category[kind]) is enabled
    assert set(system.hoomd_forcefield[-1].nlist.exclusions) == {
        "bond",
        "angle",
        "dihedral",
    }


def test_reapplication_and_atomic_failure(monkeypatch):
    system = create()
    apply(system, bonded="uff", include_impropers=False)
    original_frame, original_bundle = (
        system.hoomd_snapshot,
        system.dpd_forcefield,
    )
    before = system.assignment_report
    with pytest.raises(ValueError, match="positive"):
        apply(system, bonded="uff", bonded_scale=0)
    assert (
        system.hoomd_snapshot is original_frame
        and system.dpd_forcefield is original_bundle
    )
    assert system.assignment_report == before
    import flowermd.library.aa_system as module

    real_frame = module.create_all_atom_frame

    def fail_frame(*args, **kwargs):
        raise ValueError("injected frame failure")

    monkeypatch.setattr(module, "create_all_atom_frame", fail_frame)
    with pytest.raises(ValueError, match="injected"):
        apply(system)
    assert (
        system.hoomd_snapshot is original_frame
        and system.dpd_forcefield is original_bundle
    )
    monkeypatch.setattr(module, "create_all_atom_frame", real_frame)
    original_positions = system.gmso_system.positions.copy()
    for site in system.gmso_system.sites:
        site.position += 1 * u.nm
    system.gmso_system.remove_connection(system.gmso_system.bonds[0])
    system.system.translate([1, 1, 1])
    apply(system, bonded="sage", epsilon_weighting=False)
    assert (
        system.hoomd_snapshot is not original_frame
        and system.dpd_forcefield is not original_bundle
    )
    assert set(system.hoomd_forcefield).isdisjoint(original_bundle.hoomd_forces)
    np.testing.assert_array_equal(
        system.gmso_system.positions, original_positions
    )
    assert system.gmso_system.n_bonds == 11
    assert all(
        v == {"A": 40, "gamma": 20}
        for v in system.assignment_report["pair_coefficients"].values()
    )


def test_fixed_references_and_forbidden_mutations():
    system = create("CC")
    expected = {
        "length": 1 * u.angstrom,
        "energy": 1 * u.kcal / u.mol,
        "mass": 1 * u.amu,
    }
    assert system.reference_values == expected
    refs = system.reference_values
    refs["length"] *= 10
    assert system.reference_values == expected
    length = system.reference_length
    length *= 10
    assert system.reference_length == expected["length"]
    for name, value in (
        ("reference_values", expected),
        ("reference_length", 1 * u.nm),
        ("reference_mass", 12 * u.amu),
        ("reference_energy", 1 * u.kJ / u.mol),
        ("auto_scale", True),
    ):
        with pytest.raises(ValueError):
            setattr(system, name, value)
    with pytest.raises(ValueError, match="apply_dpd"):
        system.apply_forcefield(r_cut=3)
    with pytest.raises(ValueError, match="explicit hydrogens"):
        system.remove_hydrogens()
    apply(system, bonded="uff", include_impropers=False)
    assert (
        system.reference_values
        == expected
        == system.dpd_forcefield.reference_values
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bonded": "other"},
        {"bonded": "uff", "force_field": "openff-2.3.0.offxml"},
        {"bonded": "sage", "force_field": "openff-1.3.1.offxml"},
        {"epsilon_weighting": 1},
        {"include_torsions": 1},
        {"repulsion": -1},
        {"epsilon_weighting": False, "epsilon_reference": 1},
    ],
)
def test_invalid_apply_options(kwargs):
    system = create("CC")
    with pytest.raises(ValueError):
        apply(system, **kwargs)
    with pytest.raises(RuntimeError):
        _ = system.hoomd_snapshot


def test_openff_resource_path_object_and_sage_alias(tmp_path):
    off = pytest.importorskip("openff.toolkit")
    ff = off.ForceField("openff-1.3.1.offxml")
    path = tmp_path / "custom.offxml"
    ff.to_file(str(path))
    system = create("CC")
    for source in ("openff-1.3.1.offxml", path, ff):
        before = ff.to_string()
        apply(system, force_field=source)
        assert (
            system.assignment_report["assignment"]["source"]
            == "OpenFF SMIRNOFF"
        )
        assert ff.to_string() == before
    apply(system, bonded="sage", force_field="openff-2.3.0.offxml")
    for source in (path, off.ForceField("openff-2.3.0.offxml")):
        with pytest.raises(ValueError, match="sage fixes"):
            apply(system, bonded="sage", force_field=source)


@pytest.mark.parametrize("bonded", ["uff", "openff"])
def test_public_simulation_zero_step_and_gsd_export(tmp_path, bonded):
    system = create()
    apply(
        system,
        bonded=bonded,
        include_impropers=True,
        conservative=True,
        repulsion=0,
    )
    path = tmp_path / "owned.gsd"
    system.to_gsd(path)
    with gsd.hoomd.open(path, "r") as trajectory:
        frame = trajectory[0]
    np.testing.assert_array_equal(
        frame.particles.position, system.hoomd_snapshot.particles.position
    )
    np.testing.assert_array_equal(
        frame.dihedrals.group, system.hoomd_snapshot.dihedrals.group
    )
    simulation = Simulation.from_system(
        system,
        device=hoomd.device.CPU(),
        gsd_file_name=str(tmp_path / "sim.gsd"),
        log_file_name=str(tmp_path / "sim.txt"),
    )
    simulation.run_NVE(n_steps=0, write_at_start=False)
    assert simulation.timestep == 0
    assert np.isfinite(sum(f.energy for f in system.hoomd_forcefield))
    assert all(np.all(np.isfinite(f.forces)) for f in system.hoomd_forcefield)
    original_forces = tuple(system.hoomd_forcefield)
    apply(system, bonded="sage", epsilon_weighting=False)
    assert tuple(simulation._forcefield) == original_forces
    assert set(system.hoomd_forcefield).isdisjoint(original_forces)


def test_explicit_epsilon_reference_and_skipped_vdw():
    off = pytest.importorskip("openff.toolkit")
    unit = pytest.importorskip("openff.units").unit
    system = create("CC")
    apply(system, epsilon_reference=0.05)
    assert system.assignment_report["epsilon_reference_kcal_mol"] == 0.05
    assert system._snap_refs == system._ff_refs == system.reference_values
    original_frame = system.hoomd_snapshot
    original_report = system.assignment_report
    ff = off.ForceField("openff-2.3.0.offxml")
    for parameter in ff["vdW"].parameters:
        parameter.epsilon = float("nan") * unit.kilocalorie_per_mole
    with pytest.raises(ValueError, match="finite"):
        apply(system, force_field=ff)
    assert system.hoomd_snapshot is original_frame
    assert system.assignment_report == original_report
    apply(system, force_field=ff, epsilon_weighting=False)
    assert all(not s.atom_type.parameters for s in system.gmso_system.sites)
    assert system._snap_refs == system._ff_refs == system.reference_values


def test_array_selector_rejected_without_changing_prepared_state():
    system = create("CC")
    apply(system)
    topology = system.gmso_system
    frame = system.hoomd_snapshot
    bundle = system.dpd_forcefield
    report = system.assignment_report
    with pytest.raises(ValueError, match="bonded must be"):
        apply(system, bonded=np.array(["uff"]))
    assert system.gmso_system is topology
    assert system.hoomd_snapshot is frame
    assert system.dpd_forcefield is bundle
    assert system.assignment_report == report
