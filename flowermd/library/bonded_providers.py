"""Configured assignment providers for existing native GMSO adapters."""

from copy import deepcopy

from flowermd.internal.openff_gmso import assign_openff_parameters
from flowermd.internal.uff_gmso import assign_uff_parameters


class UFFProvider:
    """Assign all supported UFF terms, including native inversions.

    The existing UFF adapter also extracts its nonbonded parameters when
    assign_nonbonded is False. Unweighted DPD does not consume them.
    """

    def assign(self, topology, molecule, *, atom_map, assign_nonbonded=True):
        """Return native UFF parameters without execution or bonded scaling."""
        if not isinstance(assign_nonbonded, bool):
            raise ValueError("assign_nonbonded must be a bool")
        return assign_uff_parameters(
            topology, molecule, atom_map=atom_map, include_impropers=True
        )


class OpenFFProvider:
    """Configure SMIRNOFF assignment with a resource, path or ForceField.

    Configuration objects are copied on construction and on each assignment.
    Supported harmonic and periodic forms follow the existing OpenFF adapter.
    False assign_nonbonded skips vdW matching and parameter consumption.
    """

    def __init__(self, force_field="openff-2.3.0.offxml"):
        self._force_field = deepcopy(force_field)

    @property
    def force_field(self):
        """Return a copy of the configured resource or ForceField object."""
        return deepcopy(self._force_field)

    def assign(self, topology, molecule, *, atom_map, assign_nonbonded=True):
        """Return native SMIRNOFF parameters and the adapter's provenance."""
        return assign_openff_parameters(
            topology,
            molecule,
            atom_map=atom_map,
            force_field=self.force_field,
            assign_nonbonded=assign_nonbonded,
        )


class SageProvider(OpenFFProvider):
    """Configure the frozen openff-2.3.0.offxml Sage resource."""

    def __init__(self):
        super().__init__("openff-2.3.0.offxml")


def resolve_bonded_provider(bonded, force_field=None):
    """Resolve legacy strings or accept any object with a callable assign."""
    if isinstance(bonded, str):
        if bonded == "uff":
            if force_field is not None:
                raise ValueError("UFF does not accept an OpenFF force_field")
            return UFFProvider()
        if bonded == "sage":
            if force_field is not None and (
                not isinstance(force_field, str)
                or force_field != "openff-2.3.0.offxml"
            ):
                raise ValueError(
                    "sage fixes openff-2.3.0.offxml; use bonded='openff' "
                    "for another force_field"
                )
            return SageProvider()
        if bonded == "openff":
            return OpenFFProvider(
                "openff-2.3.0.offxml" if force_field is None else force_field
            )
    elif callable(getattr(bonded, "assign", None)):
        if force_field is not None:
            raise ValueError(
                "force_field is only accepted with a bonded string; "
                "configure the provider instance directly"
            )
        return bonded
    raise ValueError(
        "bonded must be uff, openff or sage, or a provider with callable assign"
    )
