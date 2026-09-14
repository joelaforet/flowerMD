"""Check native OpenFF assignment against public Toolkit and OpenMM results."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import gmso
import numpy as np
import pytest
import sympy
import unyt as u

from flowermd.internal.openff_gmso import assign_openff_parameters
from flowermd.tests.utils.test_uff_gmso import groups, inputs

openff = pytest.importorskip("openff.toolkit")
Chem = pytest.importorskip("rdkit.Chem")
unit = pytest.importorskip("openff.units").unit
RESOURCES = ("openff-1.3.1.offxml", "openff-2.3.0.offxml")


def assign(topology, molecule, atom_map, **kwargs):
    return assign_openff_parameters(
        topology, molecule, atom_map=atom_map, **kwargs
    )


def labels(molecule, force_field):
    offmol = openff.Molecule.from_rdkit(molecule, hydrogens_are_explicit=True)
    return offmol, force_field.label_molecules(offmol.to_topology())[0]


@pytest.mark.parametrize("resource", RESOURCES)
@pytest.mark.parametrize("assign_nonbonded", [True, False])
def test_resource_path_and_object_equivalence(resource, assign_nonbonded):
    from openff.toolkit.typing.engines.smirnoff import (
        get_available_force_fields,
    )

    path = next(
        Path(p)
        for p in get_available_force_fields(full_paths=True)
        if Path(p).name == resource
    )
    topology, molecule, atom_map = inputs("CC(=O)NC")
    ff = openff.ForceField(resource)
    before = ff.to_string()
    outputs = [
        assign(
            topology,
            molecule,
            atom_map,
            force_field=source,
            assign_nonbonded=assign_nonbonded,
        )
        for source in (resource, path, ff)
    ]
    assert ff.to_string() == before
    expected, name_report = outputs[0]
    for result, report in outputs:
        assert report["source"] == "OpenFF SMIRNOFF"
        assert report["assign_nonbonded"] is assign_nonbonded
        for kind in ("sites", "bonds", "angles", "dihedrals", "impropers"):
            if kind != "sites":
                assert groups(result, kind) == groups(expected, kind)
            for actual, original in zip(
                getattr(result, kind), getattr(expected, kind)
            ):
                if kind == "sites":
                    assert actual.mass == original.mass
                    actual, original = actual.atom_type, original.atom_type
                else:
                    actual, original = (
                        actual.connection_type,
                        original.connection_type,
                    )
                assert actual.name == original.name
                assert actual.tags == original.tags
                assert actual.expression == original.expression
                assert (
                    actual.independent_variables
                    == original.independent_variables
                )
                assert actual.parameters.keys() == original.parameters.keys()
                for key, value in actual.parameters.items():
                    np.testing.assert_array_equal(
                        value, original.parameters[key]
                    )
                    assert value.units == original.parameters[key].units
    path_report = outputs[1][1]
    object_report = outputs[2][1]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for report in (name_report, path_report):
        assert report["resource_sha256"] == digest
        assert report["resource_hash_scope"] == "raw OFFXML bytes"
    assert path_report["force_field"] == str(path)
    assert object_report["force_field"] == "OpenFF ForceField object"
    assert object_report["resource_sha256"] is None
    assert object_report["resource_hash_scope"] == "no resolved OFFXML resource"
    if resource == "openff-2.3.0.offxml":
        _, default_report = assign(
            topology, molecule, atom_map, assign_nonbonded=assign_nonbonded
        )
        assert default_report == name_report


@pytest.mark.parametrize("resource", RESOURCES)
def test_native_copy_coverage_metadata_and_epsilon(resource):
    original, molecule, atom_map = inputs("CC(=O)NC")
    original.sites[0].charge = 0.25 * u.elementary_charge
    original.sites[0].group = "chain"
    original.bonds[0].name = "retained bond"
    original.angles[0].name = "retained angle"
    before = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result, report = assign(original, molecule, atom_map, force_field=resource)
    ff = openff.ForceField(resource)
    offmol, raw = labels(molecule, ff)
    assert result is not original
    assert not original.is_typed()
    assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    np.testing.assert_array_equal(result.positions, original.positions)
    np.testing.assert_array_equal(result.box.lengths, original.box.lengths)
    assert result.sites[0].charge == original.sites[0].charge
    assert result.sites[0].group == "chain"
    assert result.bonds[0].name == "retained bond"
    assert result.angles[0].name == "retained angle"
    assert report["assigned_counts"] == dict(
        atoms=12, bonds=11, angles=18, proper_dihedrals=16, impropers=6
    )
    assert report["source"] == "OpenFF SMIRNOFF"
    assert report["force_field"] == resource
    if resource == "openff-2.3.0.offxml":
        assert (
            report["resource_sha256"]
            == "7a0d7195a4b717e29fe14aa2cf1bc3203e87ce4aef9cc2ef10fb5145b7edcf63"
        )
    assert report["constraint_matches"]
    assert report["constraint_treatment"] == "flexible bonded potentials"
    assert report["charges_parameterized"] is False
    assert report["versions"]["openff-toolkit"]
    for i, site in enumerate(result.sites):
        p = raw["vdW"][(i,)]
        assert site.atom_type.parameters["epsilon"].to_value(
            "kcal/mol"
        ) == pytest.approx(p.epsilon.m_as("kilocalorie_per_mole"))
        assert site.atom_type.parameters["sigma"].to_value(
            "angstrom"
        ) == pytest.approx(p.sigma.m_as("angstrom"))
        assert site.mass.to_value("amu") == pytest.approx(
            offmol.atoms[i].mass.m_as("dalton")
        )
    assert result.sites[5].atom_type is result.sites[6].atom_type
    assert result.sites[0].atom_type is not result.sites[1].atom_type
    for kind in ("sites", "bonds", "angles", "dihedrals", "impropers"):
        assert result.is_fully_typed(group=kind)
        for item in getattr(result, kind):
            potential = (
                item.atom_type if kind == "sites" else item.connection_type
            )
            assert potential.name.startswith("openff_")
            assert potential.tags["source"] == "OpenFF SMIRNOFF"
    assert any(
        len(c.dihedral_type.parameters["k"]) > 1 for c in result.dihedrals
    )


@pytest.mark.parametrize("smiles", ["CC(=O)NC", "F[C@H](Cl)C", "c1ccccc1"])
def test_permutations_and_mapped_input_preservation(smiles):
    topology, molecule, atom_map = inputs(smiles)
    base, _ = assign(topology, molecule, atom_map)
    rng = np.random.default_rng(1841)
    for _ in range(3):
        permutation = list(map(int, rng.permutation(molecule.GetNumAtoms())))
        reordered = Chem.RenumberAtoms(molecule, permutation)
        for atom in reordered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 100)
        before = reordered.ToBinary(Chem.PropertyPickleOptions.AllProps)
        result, _ = assign(topology, reordered, dict(enumerate(permutation)))
        assert reordered.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
        for i, site in enumerate(result.sites):
            assert (
                site.atom_type.parameters == base.sites[i].atom_type.parameters
            )
        np.testing.assert_allclose(
            native_energy(result, "impropers", result.positions.to_value("nm")),
            native_energy(base, "impropers", base.positions.to_value("nm")),
            atol=1e-10,
        )
        for kind in ("bonds", "angles", "dihedrals"):
            assert groups(result, kind) == groups(base, kind)


@pytest.mark.parametrize("resource", RESOURCES)
def test_unweighted_never_matches_or_reads_vdw(monkeypatch, resource):
    topology, molecule, atom_map = inputs("CC")
    ff = openff.ForceField(resource)
    handler_type = type(ff["vdW"])
    parameter_type = type(ff["vdW"].parameters[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("vdW was consumed")

    monkeypatch.setattr(handler_type, "find_matches", forbidden)
    monkeypatch.setattr(parameter_type, "epsilon", property(forbidden))
    monkeypatch.setattr(parameter_type, "sigma", property(forbidden))
    result, report = assign(
        topology, molecule, atom_map, force_field=ff, assign_nonbonded=False
    )
    assert "vdW" in ff.registered_parameter_handlers
    assert report["assign_nonbonded"] is False
    assert result.sites[0].atom_type is result.sites[1].atom_type
    for site in result.sites:
        assert site.mass > 0
        assert site.atom_type.parameters == {}
        assert site.atom_type.tags["nonbonded_assigned"] is False


def test_zero_epsilon_and_toggle_bonded_parity():
    topology, molecule, atom_map = inputs("CC")
    ff = openff.ForceField("openff-2.3.0.offxml")
    for parameter in ff["vdW"].parameters:
        parameter.epsilon = 0 * unit.kilocalorie_per_mole
    enabled, _ = assign(topology, molecule, atom_map, force_field=ff)
    disabled, _ = assign(
        topology, molecule, atom_map, force_field=ff, assign_nonbonded=False
    )
    assert all(s.atom_type.parameters["epsilon"] == 0 for s in enabled.sites)
    for kind in ("bonds", "angles", "dihedrals", "impropers"):
        assert groups(enabled, kind) == groups(disabled, kind)
        for a, b in zip(getattr(enabled, kind), getattr(disabled, kind)):
            assert a.connection_type.parameters == b.connection_type.parameters


def test_exact_improper_trefoils_and_native_serialization(tmp_path):
    topology, molecule, atom_map = inputs("CC(=O)NC")
    ff = openff.ForceField("openff-2.3.0.offxml")
    _, raw = labels(molecule, ff)
    # Reverse the two plane atoms of inferred terms, which GMSO treats as
    # equivalent, to require restoration of the exact Toolkit trefoil order.
    for improper in topology.impropers:
        c, a, b, d = improper.connection_members
        improper.connection_members = [c, b, a, d]
    result, _ = assign(topology, molecule, atom_map, force_field=ff)
    actual = dict(zip(groups(result, "impropers"), result.impropers))
    expected = set()
    for (a, c, b, d), p in raw["ImproperTorsions"].items():
        for group in ((c, a, b, d), (c, b, d, a), (c, d, a, b)):
            expected.add(group)
            potential = actual[group].improper_type
            np.testing.assert_allclose(
                potential.parameters["k"].to_value("kcal/mol"),
                [k.m_as("kilocalorie_per_mole") / 3 for k in p.k],
            )
            assert potential.tags["idivf"] == [3.0] * len(p.k)
    assert set(actual) == expected
    path = tmp_path / "sage.json"
    result.save(path)
    payload = json.loads(path.read_text())
    # GMSO 0.17 cannot reload nested potential expressions in full topologies.
    # Its native flattened potential records do preserve Fourier arrays.
    for kind, connections, type_field, cls in (
        ("dihedral_types", "dihedrals", "dihedral_type", gmso.DihedralType),
        ("improper_types", "impropers", "improper_type", gmso.ImproperType),
    ):
        saved_types = {item["id"]: item for item in payload[kind]}
        for saved, connection in zip(
            payload[connections], getattr(result, connections)
        ):
            record = deepcopy(saved_types[saved[type_field]])
            del record["id"]
            restored = cls.model_validate(record)
            original = connection.connection_type
            assert restored.expression == original.expression
            assert (
                restored.independent_variables == original.independent_variables
            )
            assert restored.tags == original.tags
            for name, value in original.parameters.items():
                np.testing.assert_array_equal(restored.parameters[name], value)
                assert restored.parameters[name].units == value.units
    assert tuple(
        tuple(p["connection_members"]) for p in payload["impropers"]
    ) == groups(result, "impropers")


def test_input_errors():
    topology, molecule, atom_map = inputs()
    for bad_map in ({}, {True: 0}, dict.fromkeys(atom_map, 0)):
        with pytest.raises(ValueError, match="bijection"):
            assign(topology, molecule, bad_map)
    with pytest.raises(ValueError, match="bool"):
        assign(topology, molecule, atom_map, assign_nonbonded=1)
    with pytest.raises(ValueError, match="explicit atoms"):
        assign(topology, Chem.MolFromSmiles("CC"), atom_map)
    typed, _ = assign(topology, molecule, atom_map)
    with pytest.raises(ValueError, match="untyped"):
        assign(typed, molecule, atom_map)
    isotope = Chem.Mol(molecule)
    isotope.GetAtomWithIdx(0).SetIsotope(13)
    with pytest.raises(ValueError, match="isotope"):
        assign(topology, isotope, atom_map)
    topology.remove_connection(topology.bonds[0])
    with pytest.raises(ValueError, match="bond graph"):
        assign(topology, molecule, atom_map)
    topology, molecule, atom_map = inputs("FC(Cl)C")
    with pytest.raises(Exception, match="unspecified stereochemistry"):
        assign(topology, molecule, atom_map)


@pytest.mark.parametrize(
    "handler", ["Bonds", "Angles", "ProperTorsions", "vdW"]
)
def test_missing_assignments(handler):
    topology, molecule, atom_map = inputs("CCC")
    ff = openff.ForceField("openff-2.3.0.offxml")
    ff.deregister_parameter_handler(handler)
    with pytest.raises(ValueError, match="missing"):
        assign(topology, molecule, atom_map, force_field=ff)


def test_unsupported_interpolation_and_forms():
    topology, molecule, atom_map = inputs("CC")
    ff = openff.ForceField("openff-2.3.0.offxml")
    _, raw = labels(molecule, ff)
    p = next(iter(raw["Bonds"].values()))
    p.k_bondorder = {1: p.k, 2: p.k}
    with pytest.raises(ValueError, match="fractional bond-order"):
        assign(topology, molecule, atom_map, force_field=ff)
    ff = openff.ForceField("openff-2.3.0.offxml")
    ff["Angles"].potential = "cosine"
    with pytest.raises(ValueError, match="unsupported Angles"):
        assign(topology, molecule, atom_map, force_field=ff)


def test_reject_removal_of_restrained_group(monkeypatch):
    topology, molecule, atom_map = inputs("CC")
    # GMSO currently exposes restraints only on angles and proper torsions.
    # Check the removal guard for an improper with such metadata as well.
    monkeypatch.setattr(
        gmso.Improper,
        "restraint",
        property(lambda self: {"k": 1}),
        raising=False,
    )
    with pytest.raises(ValueError, match="restraint"):
        assign(topology, molecule, atom_map)
    assert not topology.is_typed()


def native_energy(topology, kind, xyz):
    """Evaluate GMSO expressions with Cartesian geometric coordinates."""
    indices = {s: i for i, s in enumerate(topology.sites)}
    energy = 0.0
    for connection in getattr(topology, kind):
        group = [indices[s] for s in connection.connection_members]
        points = xyz[group]
        if kind == "bonds":
            q = np.linalg.norm(points[1] - points[0])
            units = {"k": "kJ/mol/nm**2", "r_eq": "nm"}
        elif kind == "angles":
            a, b = points[0] - points[1], points[2] - points[1]
            q = np.arccos(
                np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)
            )
            units = {"k": "kJ/mol/radian**2", "theta_eq": "radian"}
        else:
            a, b, c = (
                points[1] - points[0],
                points[2] - points[1],
                points[3] - points[2],
            )
            n1, n2 = np.cross(a, b), np.cross(b, c)
            q = np.arctan2(
                np.dot(np.cross(n1, n2), b / np.linalg.norm(b)), np.dot(n1, n2)
            )
            units = {"k": "kJ/mol", "n": "dimensionless", "phi_eq": "radian"}
        p = connection.connection_type
        variable = next(iter(p.independent_variables))
        names = list(p.parameters)
        function = sympy.lambdify([variable, *names], p.expression, "numpy")
        energy += np.sum(
            function(q, *(p.parameters[n].to_value(units[n]) for n in names))
        )
    return energy


@pytest.mark.parametrize(
    "resource, signed",
    [(RESOURCES[0], False), (RESOURCES[1], False), (RESOURCES[1], True)],
)
def test_independent_openmm_energy_and_cartesian_force_parity(resource, signed):
    mm = pytest.importorskip("openmm")
    topology, molecule, atom_map = inputs("CC(=O)NC")
    ff = openff.ForceField(resource)
    if signed:
        _, raw = labels(molecule, ff)
        for handler in ("ProperTorsions", "ImproperTorsions"):
            for p in raw[handler].values():
                p.k = [-abs(k) for k in p.k]
                p.phase = [(37 + i * 91) * unit.degree for i in range(len(p.k))]
                p.idivf = [2 + i for i in range(len(p.k))]
    result, _ = assign(topology, molecule, atom_map, force_field=ff)
    offmol, _ = labels(molecule, ff)
    offmol.partial_charges = np.zeros(offmol.n_atoms) * unit.elementary_charge
    # OpenFF/Interchange builds the reference independently of adapter output.
    ff.deregister_parameter_handler("Constraints")
    reference = ff.create_openmm_system(
        offmol.to_topology(), charge_from_molecules=[offmol]
    )
    edges = {
        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in molecule.GetBonds()
    }
    selected = {
        "bonds": mm.HarmonicBondForce(),
        "angles": mm.HarmonicAngleForce(),
        "dihedrals": mm.PeriodicTorsionForce(),
        "impropers": mm.PeriodicTorsionForce(),
    }
    for force in reference.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                selected["bonds"].addBond(*force.getBondParameters(i))
        elif isinstance(force, mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                selected["angles"].addAngle(*force.getAngleParameters(i))
        elif isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p = force.getTorsionParameters(i)
                kind = (
                    "dihedrals"
                    if all(
                        frozenset(pair) in edges for pair in zip(p[:3], p[1:4])
                    )
                    else "impropers"
                )
                selected[kind].addTorsion(*p)
    rng = np.random.default_rng(803)
    xyz = result.positions.to_value("nm") + rng.normal(
        0, 0.024, (result.n_sites, 3)
    )
    for kind, force in selected.items():
        system = mm.System()
        for _ in result.sites:
            system.addParticle(12)
        system.addForce(force)
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(
            system, integrator, mm.Platform.getPlatformByName("Reference")
        )
        context.setPositions(xyz)
        state = context.getState(getEnergy=True, getForces=True)
        expected_energy = state.getPotentialEnergy().value_in_unit(
            mm.unit.kilojoule_per_mole
        )
        expected_force = state.getForces(asNumpy=True).value_in_unit(
            mm.unit.kilojoule_per_mole / mm.unit.nanometer
        )
        assert native_energy(result, kind, xyz) == pytest.approx(
            expected_energy, rel=1e-9, abs=1e-8
        ), kind
        numerical_force = np.zeros_like(xyz)
        step = 2e-6
        for index in np.ndindex(xyz.shape):
            plus, minus = xyz.copy(), xyz.copy()
            plus[index] += step
            minus[index] -= step
            numerical_force[index] = -(
                native_energy(result, kind, plus)
                - native_energy(result, kind, minus)
            ) / (2 * step)
        np.testing.assert_allclose(
            numerical_force, expected_force, rtol=3e-6, atol=4e-4, err_msg=kind
        )
        del context, integrator


def test_missing_derived_connections_are_added():
    topology, molecule, atom_map = inputs("CC(=O)NC")
    expected, _ = assign(topology, molecule, atom_map)
    for connection in (
        *topology.angles,
        *topology.dihedrals,
        *topology.impropers,
    ):
        topology.remove_connection(connection)
    result, _ = assign(topology, molecule, atom_map)
    for kind in ("angles", "dihedrals", "impropers"):
        assert (
            {tuple(g) for g in groups(result, kind)}
            == {tuple(g) for g in groups(expected, kind)}
            if kind == "impropers"
            else len(groups(result, kind)) == len(groups(expected, kind))
        )
        assert native_energy(
            result, kind, result.positions.to_value("nm")
        ) == pytest.approx(
            native_energy(expected, kind, expected.positions.to_value("nm")),
            abs=1e-9,
        )


def test_improper_numeric_default_divisor():
    topology, molecule, atom_map = inputs("C=O")
    ff = openff.ForceField("openff-2.3.0.offxml")
    ff["ImproperTorsions"].default_idivf = 6
    result, _ = assign(topology, molecule, atom_map, force_field=ff)
    _, raw = labels(molecule, ff)
    p = next(iter(raw["ImproperTorsions"].values()))
    assert result.n_impropers == 3
    for improper in result.impropers:
        np.testing.assert_allclose(
            improper.improper_type.parameters["k"].to_value("kcal/mol"),
            [k.m_as("kilocalorie_per_mole") / 6 for k in p.k],
        )
        assert improper.improper_type.tags["idivf"] == [6.0] * len(p.k)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"k": []}, "equal nonzero lengths"),
        ({"phase": []}, "equal nonzero lengths"),
        ({"idivf": None}, "explicit idivf"),
        ({"idivf": [None]}, "explicit idivf"),
        ({"idivf": []}, "length"),
        ({"idivf": [0]}, "finite and positive"),
        ({"idivf": [float("nan")]}, "finite and positive"),
        ({"periodicity": [1.5]}, "positive integers"),
        ({"periodicity": [0]}, "positive integers"),
        ({"k": [float("inf") * unit.kilocalorie_per_mole]}, "finite"),
    ],
)
def test_invalid_fourier_data(change, message):
    from types import SimpleNamespace

    from flowermd.internal.openff_gmso import _fourier

    p = dict(
        k=[1 * unit.kilocalorie_per_mole],
        phase=[0 * unit.radian],
        periodicity=[1],
        idivf=[1],
    )
    p.update(change)
    with pytest.raises(ValueError, match=message):
        _fourier(
            SimpleNamespace(**p),
            SimpleNamespace(default_idivf="auto"),
            "ProperTorsions",
        )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_epsilon(value):
    topology, molecule, atom_map = inputs("CC")
    ff = openff.ForceField("openff-2.3.0.offxml")
    for p in ff["vdW"].parameters:
        p.epsilon = value * unit.kilocalorie_per_mole
    with pytest.raises(ValueError, match="nonnegative|finite"):
        assign(topology, molecule, atom_map, force_field=ff)
    result, _ = assign(
        topology, molecule, atom_map, force_field=ff, assign_nonbonded=False
    )
    assert all(s.atom_type.parameters == {} for s in result.sites)


@pytest.mark.parametrize(
    "coefficient, divisor", [(1e300, 1e-300), (1e-300, 1e300)]
)
def test_fourier_normalization_range(coefficient, divisor):
    from types import SimpleNamespace

    from flowermd.internal.openff_gmso import _fourier

    parameter = SimpleNamespace(
        k=[coefficient * unit.kilocalorie_per_mole],
        phase=[0 * unit.radian],
        periodicity=[1],
        idivf=[divisor],
    )
    with pytest.raises(ValueError, match="floating-point range"):
        _fourier(
            parameter, SimpleNamespace(default_idivf="auto"), "ProperTorsions"
        )
    parameter.k = [0 * unit.kilocalorie_per_mole]
    values, _ = _fourier(
        parameter, SimpleNamespace(default_idivf="auto"), "ProperTorsions"
    )
    np.testing.assert_array_equal(values["k"], [0])
