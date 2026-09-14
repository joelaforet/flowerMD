"""Check supplied chemical components independently of mBuild hierarchy."""

from copy import deepcopy

import gmso
import gsd.hoomd
import hoomd
import mbuild as mb
import numpy as np
import pytest

from flowermd import Simulation
from flowermd.library import AllAtomSystem
from flowermd.library.bonded_providers import resolve_bonded_provider
from flowermd.tests.library.test_aa_system import Chem, apply, inputs


def mixture(smiles=("CC(=O)NC", "CC(=O)NC"), hierarchy="one"):
    pieces = [inputs(s) for s in smiles]
    molecule = pieces[0][1]
    for _, mol, _ in pieces[1:]:
        molecule = Chem.CombineMols(molecule, mol)
    original = [p for compound, _, _ in pieces for p in compound.particles()]
    root = mb.Compound(name="mixture")
    children = [
        mb.Compound(name="repeated")
        for _ in range(2 if hierarchy == "split" else 1)
    ]
    root.add(children)
    created = {}
    permutation = np.random.default_rng(311).permutation(len(original))
    for old in permutation:
        p = original[old]
        new = mb.Compound(
            name=p.name,
            element=p.element,
            pos=p.pos + [0, 0, float(old >= pieces[0][0].n_particles) * 0.2],
        )
        children[old % len(children)].add(new)
        created[int(old)] = new
    offset = 0
    for compound, _, _ in pieces:
        indices = {p: i + offset for i, p in enumerate(compound.particles())}
        for a, b in compound.bonds():
            root.add_bond((created[indices[a]], created[indices[b]]))
        offset += compound.n_particles
    root.box = mb.Box(lengths=[3, 4, 5])
    sites = {p: i for i, p in enumerate(root.particles())}
    order = np.random.default_rng(84).permutation(len(original)).tolist()
    molecule = Chem.RenumberAtoms(molecule, order)
    molecule.SetProp("preserved", "caller graph")
    conformer = Chem.Conformer(len(original))
    for i, old in enumerate(order):
        conformer.SetAtomPosition(i, created[old].pos)
        molecule.GetAtomWithIdx(i).SetAtomMapNum(i + 100)
    molecule.AddConformer(conformer)
    mapping = {i: sites[created[old]] for i, old in enumerate(order)}
    return root, molecule, mapping


def partition(size, edges):
    adjacency = {i: set() for i in range(size)}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    unseen, groups = set(adjacency), set()
    while unseen:
        pending, group = [min(unseen)], set()
        while pending:
            atom = pending.pop()
            if atom not in group:
                group.add(atom)
                pending.extend(adjacency[atom] - group)
        unseen -= group
        groups.add(frozenset(group))
    return groups


@pytest.mark.parametrize("hierarchy", ["one", "split"])
def test_graph_provenance_is_owned_and_hierarchy_independent(hierarchy):
    compound, molecule, mapping = mixture(hierarchy=hierarchy)
    particles = tuple(compound.particles())
    edges = tuple(compound.bonds())
    positions = compound.xyz.copy()
    mol_bytes = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
    original_map = mapping.copy()
    system = AllAtomSystem(compound, molecule=molecule, atom_map=mapping)
    tuples = Chem.GetMolFrags(molecule)
    expected = tuple(
        dict(
            index=i,
            original_atom_indices=atoms,
            site_indices=tuple(mapping[j] for j in atoms),
        )
        for i, atoms in enumerate(tuples)
    )
    assert system.components == expected
    particle_indices = {p: i for i, p in enumerate(particles)}
    mbparts = partition(
        len(particles),
        [(particle_indices[a], particle_indices[b]) for a, b in edges],
    )
    rdparts = partition(
        len(particles),
        [
            (mapping[b.GetBeginAtomIdx()], mapping[b.GetEndAtomIdx()])
            for b in molecule.GetBonds()
        ],
    )
    native = system.gmso_system
    indices = {s: i for i, s in enumerate(native.sites)}
    gmsoparts = partition(
        len(particles),
        [tuple(indices[s] for s in b.connection_members) for b in native.bonds],
    )
    assert (
        mbparts
        == rdparts
        == gmsoparts
        == {frozenset(r["site_indices"]) for r in expected}
    )
    assert len(compound.children) == (2 if hierarchy == "split" else 1)
    if hierarchy == "split":
        assert any(a.parent is not b.parent for a, b in edges)
        assert [c.name for c in compound.children] == ["repeated", "repeated"]
    changed = system.components
    changed[0]["site_indices"] = ()
    assert system.components == expected
    apply(system, bonded="uff")
    assert system.assignment_report["construction"] == dict(
        component_count=2, components=expected
    )
    report = system.assignment_report
    report["construction"]["components"][0]["site_indices"] = ()
    system.system.translate([1, 1, 1])
    system.gmso_system.sites[0].name = "exposed change"
    apply(system, bonded="uff", epsilon_weighting=False)
    assert system.components == expected
    np.testing.assert_array_equal(
        system.gmso_system.positions.to_value("nm"), positions
    )
    assert (
        tuple(compound.particles()) == particles
        and tuple(compound.bonds()) == edges
    )
    np.testing.assert_array_equal(compound.xyz, positions)
    np.testing.assert_array_equal(compound.box.lengths, [3, 4, 5])
    assert molecule.ToBinary(Chem.PropertyPickleOptions.AllProps) == mol_bytes
    assert mapping == original_map


@pytest.mark.parametrize("bonded", ["uff", "openff", "sage"])
@pytest.mark.parametrize(
    "smiles", [("CC(=O)NC", "CC(=O)NC"), ("CC(=O)NC", "CCO")]
)
@pytest.mark.parametrize("weighting", [True, False])
def test_component_assignment_matches_separate_fragments(
    bonded, smiles, weighting
):
    compound, molecule, mapping = mixture(smiles)
    system = AllAtomSystem(compound, molecule=molecule, atom_map=mapping)
    apply(system, bonded=bonded, epsilon_weighting=weighting)
    result = system.gmso_system
    native_indices = {s: i for i, s in enumerate(result.sites)}
    clean = Chem.Mol(molecule)
    for atom in clean.GetAtoms():
        atom.SetAtomMapNum(0)
    atom_groups = []
    fragments = Chem.GetMolFrags(
        clean, asMols=True, fragsMolAtomMapping=atom_groups
    )
    source = system._construction_topology
    for fragment, atom_group in zip(fragments, atom_groups):
        global_sites = [mapping[i] for i in atom_group]
        top = gmso.Topology()
        for i in global_sites:
            top.add_site(deepcopy(source.sites[i]))
        for bond in fragment.GetBonds():
            top.add_connection(
                gmso.Bond(
                    connection_members=[
                        top.sites[bond.GetBeginAtomIdx()],
                        top.sites[bond.GetEndAtomIdx()],
                    ]
                )
            )
        assigned = resolve_bonded_provider(bonded, None).assign(
            top,
            fragment,
            atom_map=dict(enumerate(range(len(global_sites)))),
            assign_nonbonded=weighting,
        )
        expected = assigned[0]
        for i, site in enumerate(expected.sites):
            actual = result.sites[global_sites[i]]
            assert actual.mass == site.mass
            assert (
                actual.atom_type.parameters.keys()
                == site.atom_type.parameters.keys()
            )
            for key, value in site.atom_type.parameters.items():
                np.testing.assert_array_equal(
                    actual.atom_type.parameters[key], value
                )
        local_indices = {s: i for i, s in enumerate(expected.sites)}
        for kind in ("bonds", "angles", "dihedrals", "impropers"):

            def key(group):
                return group if kind == "impropers" else min(group, group[::-1])

            wanted = {
                key(
                    tuple(
                        global_sites[local_indices[s]]
                        for s in c.connection_members
                    )
                ): c.connection_type
                for c in getattr(expected, kind)
            }
            actual = {
                key(
                    tuple(native_indices[s] for s in c.connection_members)
                ): c.connection_type
                for c in getattr(result, kind)
                if native_indices[c.connection_members[0]] in global_sites
            }
            assert actual.keys() == wanted.keys()
            for group, potential in actual.items():
                for name, value in potential.parameters.items():
                    np.testing.assert_array_equal(
                        value, wanted[group].parameters[name]
                    )
    report = system.assignment_report
    if bonded == "uff":
        assert result.n_impropers == 0
        assert report["dpd"]["include_impropers"] is True
    else:
        assert report["assignment"]["components"] == system.components
        assert result.n_impropers > 0
    component = {
        i: record["index"]
        for record in system.components
        for i in record["site_indices"]
    }
    assert all(
        len({component[native_indices[s]] for s in c.connection_members}) == 1
        for c in result.connections
    )


@pytest.mark.parametrize(
    "change", ["missing_bond", "cross_bond", "map", "element"]
)
def test_mismatched_component_graphs_fail(change):
    compound, molecule, mapping = mixture()
    particles = tuple(compound.particles())
    if change == "missing_bond":
        compound.remove_bond(next(iter(compound.bonds())))
    elif change == "cross_bond":
        first, second = Chem.GetMolFrags(molecule)
        compound.add_bond(
            (particles[mapping[first[0]]], particles[mapping[second[0]]])
        )
    elif change == "map":
        first, second = Chem.GetMolFrags(molecule)
        mapping[first[0]], mapping[second[0]] = (
            mapping[second[0]],
            mapping[first[0]],
        )
    else:
        molecule.GetAtomWithIdx(0).SetAtomicNum(14)
    with pytest.raises(ValueError):
        AllAtomSystem(compound, molecule=molecule, atom_map=mapping)


@pytest.mark.parametrize("kind", ["bond", "angle"])
def test_cross_component_provider_failure_preserves_prepared_state(kind):
    compound, molecule, mapping = mixture()
    system = AllAtomSystem(compound, molecule=molecule, atom_map=mapping)
    apply(system, bonded="uff")
    before = (
        system.hoomd_snapshot,
        system.hoomd_forcefield,
        system.reference_values,
        system.assignment_report,
        system.components,
    )

    class Broken:
        def assign(self, topology, molecule, *, atom_map, assign_nonbonded):
            typed, report = resolve_bonded_provider("uff", None).assign(
                topology,
                molecule,
                atom_map=atom_map,
                assign_nonbonded=assign_nonbonded,
            )
            first, second = Chem.GetMolFrags(molecule)
            if kind == "bond":
                typed.bonds[0].connection_members = [
                    typed.sites[atom_map[first[0]]],
                    typed.sites[atom_map[second[0]]],
                ]
            else:
                members = list(typed.angles[0].connection_members)
                first_sites = {typed.sites[atom_map[i]] for i in first}
                other = second if members[1] in first_sites else first
                members[0] = typed.sites[atom_map[other[0]]]
                assert (members[0] in first_sites) != (
                    members[1] in first_sites
                )
                typed.angles[0].connection_members = members
            report["components"] = ()
            return typed, report

    with pytest.raises(ValueError):
        apply(system, bonded=Broken())
    assert system.hoomd_snapshot is before[0]
    assert system.hoomd_forcefield == before[1]
    assert system.reference_values == before[2]
    assert system.assignment_report == before[3]
    assert system.components == before[4]


def test_provider_component_report_cannot_replace_construction_truth():
    compound, molecule, mapping = mixture()
    system = AllAtomSystem(compound, molecule=molecule, atom_map=mapping)
    expected = system.components

    class ReportOnly:
        def assign(self, *args, **kwargs):
            typed, report = resolve_bonded_provider("uff", None).assign(
                *args, **kwargs
            )
            report["components"] = ()
            return typed, report

    apply(system, bonded=ReportOnly())
    assert system.components == expected
    assert system.assignment_report["construction"]["components"] == expected
    assert system.assignment_report["assignment"]["components"] == ()


def test_isolated_species_construction():
    compound = mb.Compound()
    compound.add(
        [
            mb.Compound(name="He", element="He", pos=[1, 1, 1]),
            mb.Compound(name="He", element="He", pos=[2, 2, 2]),
        ]
    )
    compound.box = mb.Box(lengths=[3, 4, 5])
    molecule = Chem.MolFromSmiles("[He].[He]")
    system = AllAtomSystem(compound, molecule=molecule, atom_map={0: 0, 1: 1})
    assert [r["site_indices"] for r in system.components] == [(0,), (1,)]
    with pytest.raises(ValueError):
        apply(system, bonded="uff")


@pytest.mark.parametrize("bonded", ["uff", "openff"])
@pytest.mark.parametrize(
    "disabled",
    [
        None,
        "include_bonds",
        "include_angles",
        "include_torsions",
        "include_impropers",
    ],
)
def test_component_frames_cpu_exclusions_and_ablations(
    tmp_path, bonded, disabled
):
    compound, molecule, mapping = mixture(("CC(=O)NC", "CCO"))
    system = AllAtomSystem(compound, molecule=molecule, atom_map=mapping)
    apply(
        system,
        bonded=bonded,
        conservative=True,
        **({disabled: False} if disabled else {}),
    )
    frame = system.hoomd_snapshot
    np.testing.assert_array_equal(frame.particles.charge, 0)
    np.testing.assert_allclose(
        (
            frame.particles.position
            + frame.particles.image * frame.configuration.box[:3]
            + frame.configuration.box[:3] / 2
        )
        / 10,
        compound.xyz,
        atol=2e-7,
    )
    path = tmp_path / "components.gsd"
    system.to_gsd(path)
    with gsd.hoomd.open(path, "r") as trajectory:
        restored = trajectory[0]
    for category in ("bonds", "angles", "dihedrals", "impropers"):
        np.testing.assert_array_equal(
            getattr(frame, category).group, getattr(restored, category).group
        )
    # Independently derive every one-, two- and three-bond endpoint pair.
    adjacency = {i: set() for i in range(molecule.GetNumAtoms())}
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].add(b)
        adjacency[b].add(a)
    expected = set()
    for first in adjacency:
        paths = [(first,)]
        for _ in range(3):
            paths = [
                (*path, neighbor)
                for path in paths
                for neighbor in adjacency[path[-1]]
                if neighbor not in path
            ]
            expected.update(
                frozenset((mapping[first], mapping[path[-1]])) for path in paths
            )
    actual = {
        frozenset((int(g[0]), int(g[-1])))
        for kind in ("bonds", "angles", "dihedrals")
        for g in getattr(frame, kind).group
    }
    assert actual == expected
    membership = {
        i: r["index"] for r in system.components for i in r["site_indices"]
    }
    assert all(len({membership[i] for i in pair}) == 1 for pair in actual)
    cross = [
        (i, j)
        for i in range(frame.particles.N)
        for j in range(i + 1, frame.particles.N)
        if membership[i] != membership[j]
    ]
    assert cross and all(frozenset(pair) not in actual for pair in cross)
    simulation = Simulation.from_system(
        system,
        device=hoomd.device.CPU(),
        gsd_file_name=str(tmp_path / "sim.gsd"),
        log_file_name=str(tmp_path / "sim.txt"),
    )
    simulation.run_NVE(n_steps=0, write_at_start=False)
    assert all(
        np.isfinite(f.energy) and np.all(np.isfinite(f.forces))
        for f in system.hoomd_forcefield
    )
    assert set(
        system.dpd_forcefield.forces_by_category["pair"][0].nlist.exclusions
    ) == {"bond", "angle", "dihedral"}
    neighbor_pairs = system.dpd_forcefield.forces_by_category["pair"][
        0
    ].nlist.pair_list
    assert any(
        membership[int(a)] != membership[int(b)] for a, b in neighbor_pairs
    )
    assert all(
        frozenset((int(a), int(b))) not in expected for a, b in neighbor_pairs
    )
    if disabled:
        category = dict(
            include_bonds="bonds",
            include_angles="angles",
            include_torsions="dihedrals",
            include_impropers="impropers",
        )[disabled]
        assert system.dpd_forcefield.forces_by_category[category] == ()
    if bonded == "uff":
        assert system.gmso_system.n_impropers == 0
    elif disabled != "include_impropers":
        assert system.dpd_forcefield.forces_by_category["impropers"]
