"""Check component mapping against independent Toolkit assignments."""

from copy import deepcopy

import gmso
import numpy as np
import pytest

from flowermd.tests.utils.test_openff_gmso import (
    RESOURCES,
    Chem,
    assign,
    labels,
    native_energy,
    openff,
    unit,
)
from flowermd.tests.utils.test_uff_gmso import inputs


def components(smiles=("CC(=O)NC", "CC(=O)NC")):
    pieces = [inputs(s) for s in smiles]
    molecule = pieces[0][1]
    for _, other, _ in pieces[1:]:
        molecule = Chem.CombineMols(molecule, other)
    sites = [site for top, _, _ in pieces for site in top.sites]
    topology = gmso.Topology()
    permutation = np.random.default_rng(42).permutation(len(sites))
    for index in permutation:
        topology.add_site(sites[index])
    for top, _, _ in pieces:
        for connection in top.connections:
            topology.add_connection(connection)
    topology.box = deepcopy(pieces[0][0].box)
    order = np.random.default_rng(73).permutation(len(sites)).tolist()
    molecule = Chem.RenumberAtoms(molecule, order)
    inverse = np.argsort(permutation)
    mapping = {i: int(inverse[old]) for i, old in enumerate(order)}
    return topology, molecule, mapping


@pytest.mark.parametrize("resource", RESOURCES)
@pytest.mark.parametrize(
    "smiles",
    [
        ("CC(=O)NC", "CC(=O)NC"),
        ("CC(=O)NC", "CCO"),
        ("F[C@H](Cl)C", "F[C@@H](Cl)C"),
    ],
)
@pytest.mark.parametrize("assign_nonbonded", [True, False])
def test_component_parameters_and_reports(resource, smiles, assign_nonbonded):
    top, molecule, atom_map = components(smiles)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(100 + atom.GetIdx())
    before = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    positions = top.positions.copy()
    top.sites[0].group = "preserved"
    top.bonds[0].name = "preserved bond"
    result, report = assign(
        top,
        molecule,
        atom_map,
        force_field=resource,
        assign_nonbonded=assign_nonbonded,
    )
    assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert not top.is_typed()
    np.testing.assert_array_equal(top.positions, positions)
    np.testing.assert_array_equal(result.positions, positions)
    assert result.sites[0].group == "preserved"
    assert result.bonds[0].name == "preserved bond"
    clean = Chem.Mol(molecule)
    for atom in clean.GetAtoms():
        atom.SetAtomMapNum(0)
    indices = []
    fragments = Chem.GetMolFrags(
        clean, asMols=True, fragsMolAtomMapping=indices
    )
    assert report["component_count"] == 2
    assert report["components"] == tuple(
        dict(
            index=i,
            original_atom_indices=tuple(group),
            site_indices=tuple(atom_map[j] for j in group),
        )
        for i, group in enumerate(indices)
    )
    expected_constraints = []
    site_index = {s: i for i, s in enumerate(result.sites)}
    for fragment, group in zip(fragments, indices):
        offmol, raw = labels(fragment, openff.ForceField(resource))
        global_sites = [atom_map[i] for i in group]
        for i, atom in enumerate(offmol.atoms):
            actual = result.sites[global_sites[i]]
            assert actual.mass.to_value("amu") == pytest.approx(
                atom.mass.m_as("dalton")
            )
            if assign_nonbonded:
                for key, units in (
                    ("epsilon", "kcal/mol"),
                    ("sigma", "angstrom"),
                ):
                    offunit = (
                        "kilocalorie_per_mole" if key == "epsilon" else units
                    )
                    assert actual.atom_type.parameters[key].to_value(
                        units
                    ) == pytest.approx(
                        getattr(raw["vdW"][(i,)], key).m_as(offunit)
                    )
            else:
                assert actual.atom_type.parameters == {}
        expected_constraints.extend(
            tuple(global_sites[i] for i in key)
            for key in raw.get("Constraints", {})
        )
        for kind, handler in (
            ("bonds", "Bonds"),
            ("angles", "Angles"),
            ("dihedrals", "ProperTorsions"),
            ("impropers", "ImproperTorsions"),
        ):

            def key(group):
                return group if kind == "impropers" else min(group, group[::-1])

            wanted = {}
            for members, parameter in raw.get(handler, {}).items():
                ordered = [members]
                if kind == "impropers":
                    a, c, b, d = members
                    ordered = [(c, a, b, d), (c, b, d, a), (c, d, a, b)]
                for members in ordered:
                    wanted[key(tuple(global_sites[i] for i in members))] = (
                        parameter
                    )
            actual = {
                key(
                    tuple(site_index[s] for s in c.connection_members)
                ): c.connection_type
                for c in getattr(result, kind)
                if site_index[c.connection_members[0]] in global_sites
            }
            assert actual.keys() == wanted.keys()
            for members, potential in actual.items():
                parameter = wanted[members]
                params = potential.parameters
                if kind == "bonds":
                    assert params["k"].to_value(
                        "kcal/mol/angstrom**2"
                    ) == pytest.approx(
                        parameter.k.m_as("kilocalorie_per_mole/angstrom**2")
                    )
                    assert params["r_eq"].to_value("angstrom") == pytest.approx(
                        parameter.length.m_as("angstrom")
                    )
                elif kind == "angles":
                    assert params["k"].to_value(
                        "kcal/mol/radian**2"
                    ) == pytest.approx(
                        parameter.k.m_as("kilocalorie_per_mole/radian**2")
                    )
                    assert params["theta_eq"].to_value(
                        "radian"
                    ) == pytest.approx(parameter.angle.m_as("radian"))
                else:
                    divisors = parameter.idivf
                    if divisors is None:
                        divisors = [3] * len(parameter.k)
                    np.testing.assert_allclose(
                        params["k"].to_value("kcal/mol"),
                        [
                            k.m_as("kilocalorie_per_mole") / divisor
                            for k, divisor in zip(parameter.k, divisors)
                        ],
                    )
                    np.testing.assert_array_equal(
                        params["n"].value, parameter.periodicity
                    )
                    np.testing.assert_allclose(
                        params["phi_eq"].to_value("radian"),
                        [phase.m_as("radian") for phase in parameter.phase],
                    )
    assert report["constraint_matches"] == tuple(expected_constraints)
    component_by_site = {
        site: i
        for i, group in enumerate(indices)
        for site in (atom_map[j] for j in group)
    }
    for connection in result.connections:
        assert (
            len(
                {
                    component_by_site[site_index[s]]
                    for s in connection.connection_members
                }
            )
            == 1
        )


@pytest.mark.parametrize("malformed", ["missing", "extra", "index"])
def test_later_component_label_failure_is_atomic(monkeypatch, malformed):
    top, molecule, atom_map = components()
    before = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    positions = top.positions.copy()
    actual = openff.ForceField.label_molecules
    calls = []

    def failing(self, topology):
        result = actual(self, topology)
        calls.append(1)
        if len(calls) == 2:
            if malformed == "missing":
                result[0]["Bonds"] = {}
            elif malformed == "extra":
                return result * 2
            else:
                key = next(iter(result[0]["Bonds"]))
                result[0]["Bonds"][(999, key[1])] = result[0]["Bonds"].pop(key)
        return result

    monkeypatch.setattr(openff.ForceField, "label_molecules", failing)
    with pytest.raises(ValueError):
        assign(top, molecule, atom_map)
    assert len(calls) == 2
    assert not top.is_typed()
    np.testing.assert_array_equal(top.positions, positions)
    assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_undefined_stereo_in_later_component_is_atomic():
    top, molecule, atom_map = components(("CCO", "FC(Cl)C"))
    with pytest.raises(Exception, match="stereo"):
        assign(top, molecule, atom_map)
    assert not top.is_typed()


@pytest.mark.parametrize(
    "resource, signed",
    [(RESOURCES[0], False), (RESOURCES[1], False), (RESOURCES[1], True)],
)
def test_component_openmm_energy_and_cartesian_force_parity(resource, signed):
    mm = pytest.importorskip("openmm")
    topology, molecule, atom_map = components()
    ff = openff.ForceField(resource)
    indices = []
    fragments = Chem.GetMolFrags(
        molecule, asMols=True, fragsMolAtomMapping=indices
    )
    offmols = [labels(fragment, ff)[0] for fragment in fragments]
    export_to_global = np.array(
        [atom_map[i] for group in indices for i in group]
    )
    if signed:
        _, raw = labels(fragments[0], ff)
        for handler in ("ProperTorsions", "ImproperTorsions"):
            for p in raw[handler].values():
                p.k = [-abs(k) for k in p.k]
                p.phase = [(37 + i * 91) * unit.degree for i in range(len(p.k))]
                p.idivf = [2 + i for i in range(len(p.k))]
    result, _ = assign(topology, molecule, atom_map, force_field=ff)
    for offmol in offmols:
        offmol.partial_charges = (
            np.zeros(offmol.n_atoms) * unit.elementary_charge
        )
    # OpenFF/Interchange builds the reference independently of adapter output.
    ff.deregister_parameter_handler("Constraints")
    reference = ff.create_openmm_system(
        openff.Topology.from_molecules(offmols),
        charge_from_molecules=[offmols[0]],
    )
    global_to_export = np.argsort(export_to_global)
    edges = {
        frozenset(
            (
                global_to_export[atom_map[b.GetBeginAtomIdx()]],
                global_to_export[atom_map[b.GetEndAtomIdx()]],
            )
        )
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
        context.setPositions(xyz[export_to_global])
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
            numerical_force[export_to_global],
            expected_force,
            rtol=3e-6,
            atol=4e-4,
            err_msg=kind,
        )
        del context, integrator


@pytest.mark.parametrize("resource", RESOURCES)
def test_every_component_skips_vdw_consumption(monkeypatch, resource):
    top, molecule, atom_map = components(("CC(=O)NC", "CCO"))
    ff = openff.ForceField(resource)

    def forbidden(*args, **kwargs):
        raise AssertionError("vdW was consumed")

    monkeypatch.setattr(type(ff["vdW"]), "find_matches", forbidden)
    for name in ("epsilon", "sigma"):
        monkeypatch.setattr(
            type(ff["vdW"].parameters[0]), name, property(forbidden)
        )
    result, report = assign(
        top, molecule, atom_map, force_field=ff, assign_nonbonded=False
    )
    assert report["component_count"] == 2
    assert all(site.atom_type.parameters == {} for site in result.sites)
