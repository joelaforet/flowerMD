"""Structural interface for assignment into native GMSO topologies."""

from collections.abc import Mapping
from typing import Any, Protocol

from gmso import Topology


class BondedParameterProvider(Protocol):
    """Assign parameters without constructing forces or changing chemistry.

    Implement assign without inheritance or registration. The inputs are a
    native untyped GMSO topology, authoritative explicit-H RDKit molecule and
    an atom-index to site-index map. Preserve site order, elements, graph,
    coordinates, box, site metadata and explicit charges. Return an assigned
    GMSO topology and a provenance mapping with a nonempty string source.

    Every assigned atom type requires a positive physical mass. True
    assign_nonbonded also requires native epsilon quantities. False imposes no
    epsilon or sigma requirement. Assign all known bonded terms regardless of
    the execution ablations. Assignment must not scale stiffnesses or place
    coordinates. The caller owns fresh input copies and copies returned data.
    Validation checks structural identity in the supplied site order. It cannot
    distinguish a permutation of coincident, chemically identical sites whose
    metadata and graph connections are also indistinguishable.
    """

    def assign(
        self,
        topology: Topology,
        molecule,
        *,
        atom_map: Mapping[int, int],
        assign_nonbonded: bool = True,
    ) -> tuple[Topology, Mapping[str, Any]]:
        """Return the assigned topology and provenance mapping."""
        ...
