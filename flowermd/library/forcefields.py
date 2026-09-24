"""All pre-defined forcefield classes for use in flowerMD."""

import itertools
import os
import time

import hoomd
import numpy as np

from flowermd.assets import FF_DIR
from flowermd.base import BaseHOOMDForcefield, BaseXMLForcefield
from flowermd.internal.all_atom_parameters import to_gsd_frame
from flowermd.internal.charges import (
    CHARGE_METHODS,
    DEFAULT_NAGL_MODEL,
    assign_partial_charges,
)
from flowermd.internal.electrostatics import (
    SmearedCoulomb,
    hoomd_charges,
    mesh_resolution,
)
from flowermd.internal.stereochemistry import (
    append_stereochemistry_dihedrals,
    build_stereochemistry_force,
    capture_stereochemistry,
)


class GAFF(BaseXMLForcefield):
    """General Amber forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/gaff.xml"):
        super(GAFF, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "The General Amber Forcefield written in foyer XML format. "
            "The XML file was obtained from the antefoyer package: "
            "https://github.com/rsdefever/antefoyer/tree/master/antefoyer"
        )


class OPLS_AA(BaseXMLForcefield):
    """OPLS All Atom forcefield class."""

    def __init__(self, name="oplsaa"):
        super(OPLS_AA, self).__init__(name=name)
        self.description = "opls-aa forcefield found in the Foyer package."


class OPLS_AA_PPS(BaseXMLForcefield):
    """OPLS All Atom for PPS molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/pps_opls.xml"):
        super(OPLS_AA_PPS, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "Based on flowermd.forcefields.OPLS_AA. "
            "Trimmed down to include only PPS parameters. "
            "One missing parameter was added manually: "
            "<Angle class1=CA class2=S class3=CA angle=1.805 k=627.6/> "
            "The equilibrium angle was determined from "
            "experimental PPS papers. "
            "The spring constant taken from the equivalent angle in GAFF."
        )


class OPLS_AA_BENZENE(BaseXMLForcefield):
    """OPLS All Atom for benzene molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/benzene_opls.xml"):
        super(OPLS_AA_BENZENE, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "Based on flowermd.forcefields.OPLS_AA. "
            "Trimmed down to include only benzene parameters."
        )


class OPLS_AA_DIMETHYLETHER(BaseXMLForcefield):
    """OPLS All Atom for dimethyl ether molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/dimethylether_opls.xml"):
        super(OPLS_AA_DIMETHYLETHER, self).__init__(
            forcefield_files=forcefield_files
        )
        self.description = (
            "Based on flowermd.forcefields.OPLS_AA. "
            "Trimmed down to include only dimethyl ether parameters."
        )


class FF_from_file(BaseXMLForcefield):
    """Forcefield class for loading a forcefield from an XML file."""

    def __init__(self, forcefield_files):
        super(FF_from_file, self).__init__(forcefield_files=forcefield_files)
        self.description = "Forcefield loaded from an XML file. "


class KremerGrestBeadSpring(BaseHOOMDForcefield):
    r"""Kremer-Grest Bead-Spring polymer coarse-grain model.

    Parameters
    ----------
    bond_k : float, required
        Spring constant in the FENE-WCA bond potential.
    bond_max : float, required
        Maximum bond length in the FENE-WCA bond potential.
    delta : float, optional, default 0.0
        The radial shift used in the FENE-WCA bond potential.
    sigma : float, optional, default 1.0
        Length scale in the 12-6 Lennard-Jones pair force.
    epsilon : float, optional, default 1.0
        Energy scale in the 12-6 Lennard-Jones pair force.
    bead_name : str, optional, default "A"
        Particle names in the bead-spring system.
    nlist : type, default hoomd.md.nlist.Cell
        A class (not an instance) of the HOOMD neighbor list
        to use for the pair force.
    nlist_buffer : float, default 0.40
        The buffer value (distance) used by the neighbor list.

    Notes
    -----
    Use this forcefield class with `flowermd.library.polymers.BeadSpring`.

    This forcefield class returns two types of interactions:

    1. 12-6 LJ pair potential with a cutoff of :math:`2^{(1/6)}\sigma`.
    2. Bond potential that includes a FENE spring and a WCA repulsive term.

    The `sigma` and `epsilon` parameters are used both for the repulsive LJ
    potential and the WCA part of the bond potential.

    """

    def __init__(
        self,
        bond_k,
        bond_max,
        radial_shift=0,
        sigma=1.0,
        epsilon=1.0,
        bead_name="A",
        nlist=hoomd.md.nlist.Cell,
        nlist_buffer=0.40,
    ):
        self.bond_k = bond_k
        self.bond_max = bond_max
        self.radial_shift = radial_shift
        self.sigma = sigma
        self.epsilon = epsilon
        self.bead_name = bead_name
        self.r_cut = 2 ** (1 / 6) * self.sigma
        self.bond_type = f"{self.bead_name}-{self.bead_name}"
        self.pair = (self.bead_name, self.bead_name)
        self.nlist = nlist
        self.nlist_buffer = nlist_buffer
        hoomd_forces = self._create_forcefield()
        super(KremerGrestBeadSpring, self).__init__(hoomd_forces)

    def _create_forcefield(self):
        """Create the hoomd force objects."""
        forces = []
        # Create pair force:
        nlist = self.nlist(buffer=self.nlist_buffer, exclusions=["bond"])
        lj = hoomd.md.pair.LJ(nlist=nlist)
        lj.params[self.pair] = dict(epsilon=self.epsilon, sigma=self.sigma)
        lj.r_cut[self.pair] = self.r_cut
        forces.append(lj)
        # Create FENE bond force:
        fene_bond = hoomd.md.bond.FENEWCA()
        fene_bond.params[self.bond_type] = dict(
            k=self.bond_k,
            r0=self.bond_max,
            epsilon=self.epsilon,
            sigma=self.sigma,
            delta=self.radial_shift,
        )
        forces.append(fene_bond)
        return forces


class BeadSpring(BaseHOOMDForcefield):
    """Bead-spring forcefield class.

    Given a dictionary of bead types, this class creates a list
    `hoomd.md.force.Force` objects to capture bonded and non-bonded
    interactions between the beads.
    For non-bonded interactions, a Lennard-Jones potential is used.
    For bonds and angles, a harmonic potential is used.
    For dihedrals, a periodic potential is used.

    Parameters
    ----------
    r_cut : float, required
        The cutoff radius for the LJ potential.
    beads : dict, required
        A dictionary of bead types. Each bead type should be a dictionary with
        the keys "epsilon" and "sigma" that correspond to the LJ parameters.
    bonds : dict, default None
        A dictionary of bond types separated by a dash. Each bond type should
        be a dictionary with the keys "r0" and "k" that correspond to the
        harmonic bond parameters.
    angles : dict, default None
        A dictionary of angle types separated by a dash. Each angle type should
        be a dictionary with the keys "t0" and "k" that correspond to the
        harmonic angle parameters.
    dihedrals : dict, default None
        A dictionary of dihedral types separated by a dash. Each dihedral type
        should be a dictionary with the keys "phi0", "k", "d", and "n" that
        correspond to the periodic dihedral parameters.
    nlist : type, default hoomd.md.nlist.Cell
        A class (not an instance) of the HOOMD neighbor list
        to use for the pair force.
    nlist_buffer : float, default 0.40
        The buffer value (distance) used by the neighbor list.
    exclusions : list, default ["bond", "1-3"]
        A list of exclusions to use in the neighbor list. The default is to
        exclude bonded and 1-3 interactions.

    Examples
    --------
    For a simple bead-spring model with two bead types A and B, the following
    code can be used:

    ::

        ff = BeadSpring(r_cut=2.5,
                beads={"A": dict(epsilon=1.0, sigma=1.0),
                       "B": dict(epsilon=2.0, sigma=2.0)},
                bonds={"A-A": dict(r0=1.1, k=300), "A-B": dict(r0=1.1, k=300)},
                angles={"A-A-A": dict(t0=2.0, k=200),
                        "A-B-A": dict(t0=2.0, k=200)},
                dihedrals={"A-A-A-A": dict(phi0=0.0, k=100, d=-1, n=1)})

    """

    def __init__(
        self,
        r_cut,
        beads,
        bonds=None,
        angles=None,
        dihedrals=None,
        nlist=hoomd.md.nlist.Cell,
        nlist_buffer=0.40,
        exclusions=["bond", "1-3"],
    ):
        self.beads = beads
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.r_cut = r_cut
        self.nlist = nlist
        self.nlist_buffer = nlist_buffer
        self.exclusions = exclusions
        hoomd_forces = self._create_forcefield()
        super(BeadSpring, self).__init__(hoomd_forces)

    def _create_forcefield(self):
        """Create the hoomd force objects."""
        forces = []
        # Create pair force:
        nlist = self.nlist(buffer=self.nlist_buffer, exclusions=self.exclusions)
        lj = hoomd.md.pair.LJ(nlist=nlist)
        bead_types = [key for key in self.beads.keys()]
        all_pairs = list(itertools.combinations_with_replacement(bead_types, 2))
        for pair in all_pairs:
            epsilon0 = self.beads[pair[0]]["epsilon"]
            epsilon1 = self.beads[pair[1]]["epsilon"]
            pair_epsilon = (epsilon0 + epsilon1) / 2

            sigma0 = self.beads[pair[0]]["sigma"]
            sigma1 = self.beads[pair[1]]["sigma"]
            pair_sigma = (sigma0 + sigma1) / 2

            lj.params[pair] = dict(epsilon=pair_epsilon, sigma=pair_sigma)
            lj.r_cut[pair] = self.r_cut
        forces.append(lj)
        # Create bond-stretching force:
        if self.bonds:
            harmonic_bond = hoomd.md.bond.Harmonic()
            for bond_type in self.bonds:
                harmonic_bond.params[bond_type] = self.bonds[bond_type]
            forces.append(harmonic_bond)
        # Create bond-bending force:
        if self.angles:
            harmonic_angle = hoomd.md.angle.Harmonic()
            for angle_type in self.angles:
                harmonic_angle.params[angle_type] = self.angles[angle_type]
            forces.append(harmonic_angle)
        # Create torsion force:
        if self.dihedrals:
            periodic_dihedral = hoomd.md.dihedral.Periodic()
            for dih_type in self.dihedrals:
                periodic_dihedral.params[dih_type] = self.dihedrals[dih_type]
            forces.append(periodic_dihedral)
        return forces


class TableForcefield(BaseHOOMDForcefield):
    """Create a set of hoomd table potentials.

    This class provides an interface for creating hoomd table
    potentials either from arrays of energy and forces, or
    from files storing the tabulated energy and forces.

    In HOOMD-Blue, table potentials are available for:

        * Pairs: `hoomd.md.pair.Table`
        * Bonds: `hoomd.md.bond.Table`
        * Angles: `hoomd.md.angle.Table`
        * Dihedrals: `hoomd.md.dihedral.Table`

    Notes
    -----
    HOOMD table potentials are initialized using arrays of energy and forces.
    It may be most convenient to store tabulated data in files,
    in that case use the `from_files` method.

    Parameters
    ----------
    pairs: dict, optional, default None
    bonds: dict, optional, default None
    angles: dict, optional, default None
    dihedrals: dict, optional, default None
    r_min: float, optional, default None
        Sets the r_min value for hoomd.md.pair.Table parameters.
    r_max : float, optional, default None
        Sets the r cutoff value for hoomd.md.pair.Table parameters.
    nlist : type, default hoomd.md.nlist.Cell
        A class (not an instance) of the HOOMD neighbor list
        to use for the pair force.
    nlist_buffer : float, default 0.40
        The buffer value (distance) used by the neighbor list.
    exclusions : list of str, optional, default ["bond", "1-3"]
        Sets exclusions for hoomd.md.pair.Table neighbor list.

        See documentation for `hoomd.md.nlist <https://hoomd-blue.readthedocs.io/en/v4.2.0/module-md-nlist.html>`_ # noqa: E501

    """

    def __init__(
        self,
        pairs=None,
        bonds=None,
        angles=None,
        dihedrals=None,
        r_min=None,
        r_cut=None,
        nlist=hoomd.md.nlist.Cell,
        nlist_buffer=0.40,
        exclusions=["bond", "1-3"],
    ):
        self.pairs = pairs
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.r_min = r_min
        self.r_cut = r_cut
        self.exclusions = exclusions
        self.nlist = nlist
        self.nlist_buffer = nlist_buffer
        self.bond_width, self.angle_width, self.dih_width = self._check_widths()
        hoomd_forces = self._create_forcefield()
        super(TableForcefield, self).__init__(hoomd_forces)

    @classmethod
    def from_files(
        cls,
        pairs=None,
        bonds=None,
        angles=None,
        dihedrals=None,
        exclusions=["bond", "1-3"],
        nlist=hoomd.md.nlist.Cell,
        nlist_buffer=0.40,
        **kwargs,
    ):
        """Create a table forefield from files.

        Parameters
        ----------
        pairs: dict, optional, default None
            Dictionary with keys of pair type and keys of file path
        bonds: dict, optional, default None
            Dictionary with keys of bond type and keys of file path
        angles: dict, optional, default None
            Dictionary with keys of angle type and keys of file path
        dihedrals: dict, optional, default None
            Dictionary with keys of dihedral type and keys of file path
        ``**kwargs`` : keyword arguments
            Key word arguments passed to `numpy.genfromtxt` or `numpy.load`

        Notes
        -----
        The parameters must use a `{"type": "file_path"}` mapping.

        Following HOOMD conventions, pair types must be given as a `tuple`
        of particles types while bonds, angles and dihedrals
        are given as a `str` of particle types separated by dashes.

        Example
        -------
        .. code-block:: python

            table_forcefield = TableForcefield.from_files(
                pairs = {
                    ("A", "A"): "A_pairs.txt"
                    ("B", "B"): "B_pairs.txt"
                    ("A", "B"): "AB_pairs.txt"
                },
                bonds = {"A-A": "A_bonds.txt", "B-B": "B_bonds.txt"},
                angles = {"A-A-A": "A_angles.txt", "B-B-B": "B_angles.txt"},
            )

        Warning
        -------
        It is assumed that the structure of the files are:
            * Column 1: Independent variable (e.g. distance, length, angle)
            * Column 2: Energy
            * Column 3: Force

        """

        def _load_file(file, **kwargs):
            """Call the correct numpy method."""
            if not os.path.exists(file):
                raise ValueError(f"Unable to load file {file}")
            if file.split(".")[-1] in ["txt", "csv"]:
                return np.genfromtxt(file, **kwargs)
            elif file.split(".")[-1] in ["npy", "npz"]:
                return np.load(file, **kwargs)
            else:
                raise ValueError(
                    "Creating table forcefields from files only supports "
                    "using numpy.genfromtxt() with .txt, and .csv files, "
                    "or using numpy.load() with .npy or npz files."
                )

        # Read pair files
        pair_dict = dict()
        pair_r_min = set()
        pair_r_max = set()
        if pairs:
            for pair_type in pairs:
                table = _load_file(pairs[pair_type], **kwargs)
                r = table[:, 0]
                pair_r_min.add(r[0])
                pair_r_max.add(r[-1])
                pair_dict[pair_type] = dict()
                pair_dict[pair_type]["U"] = table[:, 1]
                pair_dict[pair_type]["F"] = table[:, 2]
            if len(pair_r_min) != len(pair_r_max) != 1:
                raise ValueError(
                    "All pair files must have the same r-range values"
                )
        # Read bond files
        bond_dict = dict()
        if bonds:
            for bond_type in bonds:
                table = _load_file(bonds[bond_type], **kwargs)
                r = table[:, 0]
                r_min = r[0]
                r_max = r[-1]
                bond_dict[bond_type] = dict()
                bond_dict[bond_type]["r_min"] = r_min
                bond_dict[bond_type]["r_max"] = r_max
                bond_dict[bond_type]["U"] = table[:, 1]
                bond_dict[bond_type]["F"] = table[:, 2]
        # Read angle files
        angle_dict = dict()
        if angles:
            for angle_type in angles:
                table = _load_file(angles[angle_type], **kwargs)
                thetas = table[:, 0]
                if thetas[0] != 0 or not np.allclose(
                    thetas[-1], np.pi, atol=1e-5
                ):
                    raise ValueError(
                        "Angle values must be evenly spaced and "
                        "range from 0 to Pi."
                    )
                angle_dict[angle_type] = dict()
                angle_dict[angle_type]["U"] = table[:, 1]
                angle_dict[angle_type]["F"] = table[:, 2]
        # Read dihedral files
        dih_dict = dict()
        if dihedrals:
            for dih_type in dihedrals:
                table = _load_file(dihedrals[dih_type], **kwargs)
                thetas = table[:, 0]
                if not np.allclose(
                    thetas[0], -np.pi, atol=1e-5
                ) or not np.allclose(thetas[-1], np.pi, atol=1e-5):
                    raise ValueError(
                        "Dihedral angle values must be evenly spaced and "
                        "range from -Pi to Pi."
                    )
                dih_dict[dih_type] = dict()
                dih_dict[dih_type]["U"] = table[:, 1]
                dih_dict[dih_type]["F"] = table[:, 2]

        return cls(
            pairs=pair_dict,
            bonds=bond_dict,
            angles=angle_dict,
            dihedrals=dih_dict,
            r_min=list(pair_r_min)[0],
            r_cut=list(pair_r_max)[0],
            exclusions=exclusions,
            nlist=nlist,
            nlist_buffer=nlist_buffer,
        )

    def _create_forcefield(self):
        forces = []
        # Create pair forces
        if self.pairs:
            nlist = self.nlist(
                buffer=self.nlist_buffer, exclusions=self.exclusions
            )
            pair_table = hoomd.md.pair.Table(
                nlist=nlist, default_r_cut=self.r_cut
            )
            for pair_type in self.pairs:
                U = self.pairs[pair_type]["U"]
                F = self.pairs[pair_type]["F"]
                if len(U) != len(F):
                    raise ValueError(
                        "The energy and force arrays are not the same size."
                    )
                pair_table.params[tuple(pair_type)] = dict(
                    r_min=self.r_min, U=U, F=F
                )
            forces.append(pair_table)
        # Create bond forces
        if self.bonds:
            bond_table = hoomd.md.bond.Table(width=self.bond_width)
            for bond_type in self.bonds:
                bond_table.params[tuple(bond_type)] = dict(
                    r_min=self.bonds[bond_type]["r_min"],
                    r_max=self.bonds[bond_type]["r_max"],
                    U=self.bonds[bond_type]["U"],
                    F=self.bonds[bond_type]["F"],
                )
            forces.append(bond_table)
        # Create angle forces
        if self.angles:
            angle_table = hoomd.md.angle.Table(width=self.angle_width)
            for angle_type in self.angles:
                angle_table.params[angle_type] = dict(
                    U=self.angles[angle_type]["U"],
                    tau=self.angles[angle_type]["F"],
                )
            forces.append(angle_table)
        # Create dihedral forces
        if self.dihedrals:
            dih_table = hoomd.md.dihedral.Table(width=self.dih_width)
            for dih_type in self.dihedrals:
                dih_table.params[dih_type] = dict(
                    U=self.dihedrals[dih_type]["U"],
                    tau=self.dihedrals[dih_type]["F"],
                )
            forces.append(dih_table)
        return forces

    def _check_widths(self):
        """Check number of points for bonds, pairs and angles."""
        bond_width = None
        for bond_type in self.bonds:
            new_width = len(self.bonds[bond_type]["U"])
            if bond_width is None:
                bond_width = new_width
            else:
                if new_width != bond_width:
                    raise ValueError(
                        "All bond types must have the same "
                        "number of points for table energies and forces."
                    )

        angle_width = None
        for angle_type in self.angles:
            new_width = len(self.angles[angle_type]["U"])
            if angle_width is None:
                angle_width = new_width
            else:
                if new_width != angle_width:
                    raise ValueError(
                        "All angle types must have the same "
                        "number of points for table energies and forces."
                    )

        dih_width = None
        for dih_type in self.dihedrals:
            new_width = len(self.dihedrals[dih_type]["U"])
            if dih_width is None:
                dih_width = new_width
            else:
                if new_width != dih_width:
                    raise ValueError(
                        "All dihedral types must have the same "
                        "number of points for table energies and forces."
                    )
        return bond_width, angle_width, dih_width


class EllipsoidForcefield(BaseHOOMDForcefield):
    """A forcefield for modeling anisotropic bead polymers.

    Notes
    -----
    This is designed to be used with `flowermd.library.polymers.EllipsoidChain`
    and uses ghost particles of type "A" and "B" for intra-molecular
    interactions of bonds and two-body angles.
    Ellipsoid centers (type "R") are used in inter-molecular pair interations.

    The set of interactions are:
    1. `hoomd.md.bond.Harmonic`: Models ellipsoid bonds as tip-to-tip bonds
    2. `hoomd.md.angle.Harmonic`: Models angles of two neighboring ellipsoids.
    3. `hoomd.md.pair.aniso.GayBerne`" Model pair interactions between beads.

    Parameters
    ----------
    epsilon : float, required
        energy
    lpar: float, required
        Semi-axis length of the ellipsoid along the major axis.
    lperp : float, required
        Semi-axis length of the ellipsoid along the minor axis.
    r_cut : float, required
        Cut off radius for pair interactions
    angle_k : float, required
        Spring constant in harmonic angle.
    angle_theta0: float, required
        Equilibrium angle between 2 consecutive beads.
    bond_k : float, required
        Spring constant in harmonic bond.
    bond_r0: float, required
        Equilibrium distance between 2 ellipsoid tips.
    nlist : type, default hoomd.md.nlist.Cell
        A class (not an instance) of the HOOMD neighbor list
        to use for the pair force.
    nlist_buffer : float, default 0.40
        The buffer value (distance) used by the neighbor list.

    """

    def __init__(
        self,
        epsilon,
        lpar,
        lperp,
        r_cut,
        angle_k=None,
        angle_theta0=None,
        bond_k=100,
        bond_r0=0.1,
        nlist=hoomd.md.nlist.Cell,
        nlist_buffer=0.40,
    ):
        self.epsilon = epsilon
        self.lperp = lperp
        self.lpar = lpar
        self.r_cut = r_cut
        self.angle_k = angle_k
        self.angle_theta0 = angle_theta0
        self.bond_k = bond_k
        self.bond_r0 = bond_r0
        self.nlist = nlist
        self.nlist_buffer = nlist_buffer
        hoomd_forces = self._create_forcefield()
        super(EllipsoidForcefield, self).__init__(hoomd_forces)

    def _create_forcefield(self):
        forces = []
        # Bonds
        bond = hoomd.md.bond.Harmonic()
        bond.params["T-T"] = dict(k=self.bond_k, r0=self.bond_r0)
        bond.params["A-X"] = dict(k=0, r0=0)
        forces.append(bond)
        # Angles
        if all([self.angle_k, self.angle_theta0]):
            angle = hoomd.md.angle.Harmonic()
            angle.params["X-A-X"] = dict(k=self.angle_k, t0=self.angle_theta0)
            angle.params["A-X-A"] = dict(k=0, t0=0)
            forces.append(angle)
        # Gay-Berne Pairs
        nlist = self.nlist(buffer=self.nlist_buffer, exclusions=["body"])
        gb = hoomd.md.pair.aniso.GayBerne(nlist=nlist, default_r_cut=self.r_cut)
        gb.params[("X", "X")] = dict(
            epsilon=self.epsilon, lperp=self.lperp, lpar=self.lpar
        )
        # Add zero pairs
        for pair in [
            ("R", "R"),
            ("T", "T"),
            ("T", "R"),
            ("A", "A"),
            ("A", "X"),
            ("A", "T"),
            ("A", "R"),
            ("X", "R"),
            ("X", "T"),
        ]:
            gb.params[pair] = dict(epsilon=0.0, lperp=0.0, lpar=0.0)
            gb.params[pair].r_cut = 0.0
        forces.append(gb)
        return forces


class EllipsoidFF_DPD(BaseHOOMDForcefield):
    """A DPD forcefield on anisotropic rigid bodies.

    Notes
    -----
    This is designed to be used with `flowermd.library.polymers.EllipsoidChainRand`
    and uses ghost particles of type "A" for intra-molecular
    interactions of bonds and two-body angles.
    Ellipsoid centers (type "R") are used in inter-molecular pair interations.
    The hoomd DPD forcefield is being used here with spherical rigid bodies of the flowerMD ellipsoid model for anchor points and orientation vectors to build back-mapping tools.
    Spherical dimensions of lpar = lperp = 0.5 are recommended, since DPD does not consider anisotropy and for mapping to number density.

    The set of interactions are:
    1. `hoomd.md.bond.Harmonic`: Models ellipsoid bonds as tip-to-tip bonds
    2. `hoomd.md.angle.Harmonic`: Models angles of two neighboring ellipsoids.
    3. `hoomd.md.pair.DPD`" Model pair interactions between beads. Does not consider anisotropic shapes.

    Parameters
    ----------
    epsilon : float, required
        energy
    lpar: float, required
        Semi-axis length of the ellipsoid along the major axis.
    lperp : float, required
        Semi-axis length of the ellipsoid along the minor axis.
    A : int, required
        DPD pair-wise drag force coefficient
    gamma : int, required
        DPD pair-wise random force coefficient
    kT : float, required
        Temperature used in pair-wise drag force
    r_cut : float, required
        Cut off radius for pair interactions
    angle_k : float, required
        Spring constant in harmonic angle.
    angle_theta0: float, required
        Equilibrium angle between 2 consecutive beads.
    bond_k : float, required
        Spring constant in harmonic bond.
    bond_r0: float, required
        Equilibrium distance between 2 ellipsoid tips.
    nlist : type, default hoomd.md.nlist.Cell
        A class (not an instance) of the HOOMD neighbor list
        to use for the pair force.
    nlist_buffer : float, default 0.40
        The buffer value (distance) used by the neighbor list.

    """

    def __init__(
        self,
        epsilon,
        lpar,
        lperp,
        A,
        gamma,
        kT,
        r_cut,
        angle_k=None,
        angle_theta0=None,
        bond_k=100,
        bond_r0=1.1,
        nlist=hoomd.md.nlist.Cell,
        nlist_buffer=0.40,
    ):
        self.epsilon = epsilon
        self.lpar = lpar
        self.lperp = lperp
        self.gamma = gamma
        self.A = A
        self.kT = kT
        self.r_cut = r_cut
        self.angle_k = angle_k
        self.angle_theta0 = angle_theta0
        self.bond_k = bond_k
        self.bond_r0 = bond_r0
        self.nlist = nlist
        self.nlist_buffer = nlist_buffer
        hoomd_forces = self._create_forcefield()
        super(EllipsoidFF_DPD, self).__init__(hoomd_forces)

    def _create_forcefield(self):
        forces = []
        # Bonds
        bond = hoomd.md.bond.Harmonic()
        bond.params["T-T"] = dict(k=self.bond_k, r0=self.bond_r0)
        bond.params["A-X"] = dict(k=0, r0=0)
        forces.append(bond)
        # Angles
        if all([self.angle_k, self.angle_theta0]):
            angle = hoomd.md.angle.Harmonic()
            angle.params["X-A-X"] = dict(k=self.angle_k, t0=self.angle_theta0)
            angle.params["A-X-A"] = dict(k=0, t0=0)
            forces.append(angle)
        # DPD Pairs
        nlist = self.nlist(buffer=self.nlist_buffer, exclusions=["body"])
        dpd = hoomd.md.pair.DPD(
            nlist=nlist, kT=self.kT, default_r_cut=self.r_cut
        )
        dpd.params[("X", "X")] = dict(A=self.A, gamma=self.gamma)
        # Add zero pairs
        for pair in [
            ("R", "R"),
            ("T", "T"),
            ("T", "R"),
            ("A", "A"),
            ("A", "X"),
            ("A", "T"),
            ("A", "R"),
            ("X", "R"),
            ("X", "T"),
        ]:
            dpd.params[pair] = dict(A=0, gamma=0.1)
            dpd.params[pair].r_cut = 0.0
        forces.append(dpd)
        return forces


class AllAtomDPD(BaseHOOMDForcefield):
    """All-atom dissipative particle dynamics force field for melt initialization.

    Bonded terms come from a standard atomistic force field, either UFF
    through RDKit or a SMIRNOFF force field (Sage) through OpenFF, scaled by
    `bonded_scale`. The nonbonded term is `hoomd.md.pair.DPD`, whose
    repulsion ``A`` and friction ``gamma`` are weighted per pair by the
    source's Lennard-Jones well depths,
    ``A_ij = A * sqrt(eps_i * eps_j) / eps_max``. This is the PhantomWalk
    all-atom initializer's interaction model: soft repulsion that lets
    overlapping chains pass through each other while the bonded surrogate
    keeps bond lengths, angles and torsions near their force-field minima.

    Units are Angstrom, kcal/mol and amu. Because interaction types are
    named by their coefficients rather than by element pairs, this class
    also builds the matching `gsd.hoomd.Frame`, exposed as `frame`. Pass
    ``initial_state=ff.frame, forcefield=ff.hoomd_forces`` to `Simulation`.

    Parameters
    ----------
    compound : mbuild.Compound, required
        All-atom compound with elements on every particle and a periodic
        ``box``. Every disconnected child is one molecule.
    bonded : {"uff", "openff"}, default "uff"
        Source of the bonded parameters and of the per-type epsilons.
        ``"openff"`` requires openff-toolkit and openff-interchange.
    A : float, default 1250.0
        DPD repulsion coefficient before epsilon weighting (kcal/mol/A).
    gamma : float, default 200.0
        DPD friction coefficient before epsilon weighting.
    kT : float, default 1.0
        DPD thermostat temperature in energy units (kcal/mol).
    r_cut : float, default 3.5
        DPD cutoff in Angstrom.
    bonded_scale : float, default 30.0
        Multiplier applied to every bonded ``k``.
    epsilon_weighting : bool, default True
        Weight ``A`` and ``gamma`` by ``sqrt(eps_i eps_j) / eps_max``.
        When False every pair uses ``A`` and ``gamma`` unchanged.
    include_bonds, include_angles, include_dihedrals, include_impropers : bool
        Include the corresponding bonded force. The topology stays in the
        frame either way, so neighbor-list exclusions do not change; only
        the energy term is dropped. UFF provides no impropers.
    conservative : bool, default False
        Use `hoomd.md.pair.DPDConservative` (no friction, no noise) instead
        of `hoomd.md.pair.DPD`. Then `kT` and `gamma` are unused.
    force_field : str, default "openff-2.3.0.offxml"
        SMIRNOFF force field when ``bonded="openff"``.
    protect_stereochemistry : bool, default True
        Record the handedness of every tetrahedral stereocenter and add a
        native periodic-dihedral restraint (see
        `flowermd.internal.stereochemistry`) that keeps it during DPD and
        FIRE. Has no effect on molecules without stereocenters.
    stereo_k : float, default 30000.0
        Restraint strength in kcal/mol. Not multiplied by `bonded_scale`.
    stereo_planar_tolerance : float, default 0.05
        Normalized-volume threshold below which a center counts as planar.
    nlist : type, default hoomd.md.nlist.Cell
        Neighbor list class for the pair force.
    nlist_buffer : float, default 0.4
        Neighbor list buffer in Angstrom.
    exclusions : list, default ["bond", "angle", "dihedral"]
        Neighbor-list exclusions (1-2, 1-3 and 1-4 pairs).
    charges : {None, "formal", "gasteiger", "nagl"} or array-like, default None
        Partial charges in e. A string assigns them per distinct molecule
        (see `flowermd.internal.charges`); an array gives one charge per
        particle. None leaves every charge at zero. Charges alone add no
        force; set `electrostatics` to use them.
    electrostatics : {None, "smeared"}, default None
        ``"smeared"`` adds Gaussian-smeared Coulomb interactions computed on
        a PPPM mesh (see `flowermd.internal.electrostatics`). They are
        bounded as atoms pass through each other, so they can be used with
        the soft DPD core. Requires `charges` and a neutral system.
    charge_smearing : float, default 2.0
        Standard deviation of each Gaussian charge cloud in Angstrom. The
        pair energy is ``332.06 q_i q_j erf(r / (2 sigma)) / r`` kcal/mol.
    charge_scale : float, default 1.0
        Multiplier applied to every charge in the electrostatic term.
    pppm_order : int, default 5
        PPPM charge assignment order.
    pppm_resolution : tuple of int, optional
        PPPM grid; by default the spacing is at most `charge_smearing`,
        which reproduces the smeared forces to about 1 %.
    nagl_model : str, default "openff-gnn-am1bcc-1.0.0.pt"
        NAGL model used when ``charges="nagl"``.

    Attributes
    ----------
    hoomd_forces : list
        The HOOMD forces, bonded terms first and the pair force last.
    frame : gsd.hoomd.Frame
        Initial state with types matching the forces.
    parameters : flowermd.internal.all_atom_parameters.AllAtomParameters
        The numeric tables the forces were built from.
    forces_by_role : dict
        ``{"bond": force, "angle": ..., "dihedral": ..., "improper": ...,
        "pair": ...}`` for the forces that were created.
    openff_topology : openff.toolkit.Topology or None
        The OpenFF topology used for parameter assignment when
        ``bonded="openff"``, for building a charged Interchange downstream.
        Its molecules carry `charges` when those were assigned.
    stereo_reference : flowermd.internal.stereochemistry.StereoReference or None
        The recorded stereocenters when `protect_stereochemistry` is True.
    charges : numpy.ndarray
        One partial charge per particle in e (zeros when `charges` is None).
        The frame stores them in HOOMD units, scaled by `charge_scale`.
    charge_method : str or None
        ``"formal"``, ``"gasteiger"``, ``"nagl"``, ``"user"`` or None.
    timings : dict
        Wall time in seconds for ``parameterization``, ``charges`` and
        ``setup`` (frame and force construction).

    """

    def __init__(
        self,
        compound,
        bonded="uff",
        A=1250.0,
        gamma=200.0,
        kT=1.0,
        r_cut=3.5,
        bonded_scale=30.0,
        epsilon_weighting=True,
        include_bonds=True,
        include_angles=True,
        include_dihedrals=True,
        include_impropers=True,
        conservative=False,
        force_field="openff-2.3.0.offxml",
        protect_stereochemistry=True,
        stereo_k=30000.0,
        stereo_planar_tolerance=0.05,
        nlist=hoomd.md.nlist.Cell,
        nlist_buffer=0.4,
        exclusions=["bond", "angle", "dihedral"],
        charges=None,
        electrostatics=None,
        charge_smearing=2.0,
        charge_scale=1.0,
        pppm_order=5,
        pppm_resolution=None,
        nagl_model=DEFAULT_NAGL_MODEL,
    ):
        if bonded not in ("uff", "openff"):
            raise ValueError("bonded must be 'uff' or 'openff'.")
        if electrostatics not in (None, "smeared"):
            raise ValueError("electrostatics must be None or 'smeared'.")
        if electrostatics and charges is None:
            raise ValueError("electrostatics='smeared' needs charges.")
        if isinstance(charges, str) and charges not in CHARGE_METHODS:
            raise ValueError(f"charges must be one of {CHARGE_METHODS}.")
        if charge_smearing <= 0:
            raise ValueError("charge_smearing must be positive.")
        for name, value in (
            ("A", A),
            ("kT", kT),
            ("r_cut", r_cut),
            ("bonded_scale", bonded_scale),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if gamma < 0:
            raise ValueError("gamma cannot be negative.")
        if protect_stereochemistry and stereo_k <= 0:
            raise ValueError("stereo_k must be positive.")
        self.protect_stereochemistry = protect_stereochemistry
        self.stereo_k = stereo_k
        self.stereo_planar_tolerance = stereo_planar_tolerance
        self.stereo_reference = None
        self.timings = {}
        self.bonded = bonded
        self.A = A
        self.gamma = gamma
        self.kT = kT
        self.r_cut = r_cut
        self.bonded_scale = bonded_scale
        self.epsilon_weighting = epsilon_weighting
        self.include_bonds = include_bonds
        self.include_angles = include_angles
        self.include_dihedrals = include_dihedrals
        self.include_impropers = include_impropers
        self.conservative = conservative
        self.force_field = force_field
        self.nlist = nlist
        self.nlist_buffer = nlist_buffer
        self.exclusions = list(exclusions)
        self.openff_topology = None
        self.electrostatics = electrostatics
        self.charge_smearing = charge_smearing
        self.charge_scale = charge_scale
        self.pppm_order = pppm_order
        self.pppm_resolution = pppm_resolution
        self.nagl_model = nagl_model
        started = time.perf_counter()
        if bonded == "uff":
            from flowermd.internal.uff_parameters import parameterize_uff

            self.parameters = parameterize_uff(compound)
        else:
            from flowermd.internal.openff_parameters import (
                parameterize_openff,
            )

            self.parameters, self.openff_topology = parameterize_openff(
                compound, force_field=force_field
            )
        if protect_stereochemistry:
            self.stereo_reference = capture_stereochemistry(
                compound, planar_tolerance=stereo_planar_tolerance
            )
        parameterized = time.perf_counter()
        n_particles = self.parameters.n_particles
        if charges is None:
            self.charge_method = None
            self.charges = np.zeros(n_particles)
        elif isinstance(charges, str):
            self.charge_method = charges
            self.charges = assign_partial_charges(
                compound, charges, nagl_model=nagl_model
            )
        else:
            self.charge_method = "user"
            self.charges = np.asarray(charges, dtype=float)
            if self.charges.shape != (n_particles,):
                raise ValueError(
                    f"charges needs one value per particle ({n_particles})."
                )
        if electrostatics and abs(self.charges.sum()) > 1e-6:
            raise ValueError(
                f"The system carries a net charge of {self.charges.sum():.4f} "
                "e. Add counterions; smeared electrostatics needs a neutral "
                "system."
            )
        if self.openff_topology is not None and self.charge_method:
            # Carry the charges to the topology used for a Sage hand-off.
            from openff.units import unit as offunit

            offset = 0
            for molecule in self.openff_topology.molecules:
                n = molecule.n_atoms
                molecule.partial_charges = (
                    self.charges[offset : offset + n]
                    * offunit.elementary_charge
                )
                offset += n
        charged = time.perf_counter()
        self.frame = to_gsd_frame(self.parameters)
        self.frame.particles.charge = hoomd_charges(self.charges, charge_scale)
        if self.stereo_reference is not None and self.stereo_reference.centers:
            append_stereochemistry_dihedrals(self.frame, self.stereo_reference)
        self.forces_by_role = {}
        hoomd_forces = self._create_forcefield()
        self.timings = {
            "parameterization": parameterized - started,
            "charges": charged - parameterized,
            "setup": time.perf_counter() - charged,
        }
        super(AllAtomDPD, self).__init__(hoomd_forces)

    @property
    def net_charge(self):
        """Total charge of the system in e."""
        return float(self.charges.sum())

    @property
    def stereo_centers(self):
        """Number of protected tetrahedral centers (0 when unprotected)."""
        if self.stereo_reference is None:
            return 0
        return len(self.stereo_reference.centers)

    def conservative_forces(self):
        """The same bonded and restraint forces with a conservative DPD pair.

        `hoomd.md.pair.DPDConservative` keeps only the repulsion ``A``, no
        friction or noise, which is what a minimizer such as FIRE needs.
        The bonded force objects are shared with `hoomd_forces`, so attach
        only one of the two lists to an integrator at a time.
        """
        pair = self.forces_by_role["pair"]
        return [f for f in self.hoomd_forces if f is not pair] + [
            self._pair_force(conservative=True)
        ]

    def _scaled(self, params):
        return {
            name: {**values, "k": self.bonded_scale * values["k"]}
            for name, values in params.items()
        }

    def _create_forcefield(self):
        p = self.parameters
        forces = []
        if self.include_bonds and p.bonds:
            force = hoomd.md.bond.Harmonic()
            for name, values in self._scaled(p.bond_params).items():
                force.params[name] = values
            self.forces_by_role["bond"] = force
            forces.append(force)
        if self.include_angles and p.angles:
            force = hoomd.md.angle.Harmonic()
            for name, values in self._scaled(p.angle_params).items():
                force.params[name] = values
            self.forces_by_role["angle"] = force
            forces.append(force)
        if self.include_dihedrals and p.dihedrals:
            force = hoomd.md.dihedral.Periodic()
            for name, values in self._scaled(p.dihedral_params).items():
                force.params[name] = values
            self.forces_by_role["dihedral"] = force
            forces.append(force)
        if self.include_impropers and p.impropers:
            # SMIRNOFF impropers are periodic torsions over the improper's
            # own atom order, so the same functional form applies.
            force = hoomd.md.improper.Periodic()
            for name, values in self._scaled(p.improper_params).items():
                force.params[name] = {
                    "k": values["k"],
                    "d": values["d"],
                    "n": values["n"],
                    "chi0": values["phi0"],
                }
            self.forces_by_role["improper"] = force
            forces.append(force)

        stereo_types = (
            self.stereo_reference.type_names if self.stereo_centers else ()
        )
        if stereo_types and "dihedral" in self.forces_by_role:
            # A dihedral force must cover every dihedral type in the state.
            for name in stereo_types:
                self.forces_by_role["dihedral"].params[name] = {
                    "k": 0.0,
                    "d": -1,
                    "n": 1,
                    "phi0": 0.0,
                }
        if stereo_types:
            restraint = build_stereochemistry_force(
                self.stereo_reference,
                self.stereo_k,
                dihedral_types=list(p.dihedral_params),
            )
            self.forces_by_role["stereochemistry"] = restraint
            forces.append(restraint)

        if self.electrostatics == "smeared":
            electrostatics = self._electrostatic_force()
            self.forces_by_role["electrostatics"] = electrostatics
            forces.append(electrostatics)

        pair = self._pair_force(conservative=self.conservative)
        self.forces_by_role["pair"] = pair
        forces.append(pair)
        return forces

    def _electrostatic_force(self):
        sigma = self.charge_smearing
        if self.pppm_resolution is None:
            self.pppm_resolution = mesh_resolution(
                self.parameters.box_lengths_a, 1.0 / (2.0 * sigma)
            )
        nlist = self.nlist(buffer=self.nlist_buffer, exclusions=self.exclusions)
        return SmearedCoulomb(
            nlist=nlist,
            resolution=tuple(int(n) for n in self.pppm_resolution),
            order=self.pppm_order,
            sigma=sigma,
        )

    def _pair_force(self, conservative):
        p = self.parameters
        nlist = self.nlist(buffer=self.nlist_buffer, exclusions=self.exclusions)
        if conservative:
            pair = hoomd.md.pair.DPDConservative(
                nlist=nlist, default_r_cut=self.r_cut
            )
        else:
            pair = hoomd.md.pair.DPD(
                nlist=nlist, kT=self.kT, default_r_cut=self.r_cut
            )
        types = list(dict.fromkeys(p.particle_types))
        for first, second in itertools.combinations_with_replacement(types, 2):
            if self.epsilon_weighting:
                weight = (
                    np.sqrt(
                        p.particle_epsilons[first] * p.particle_epsilons[second]
                    )
                    / p.epsilon_ref
                )
            else:
                weight = 1.0
            values = {"A": self.A * weight}
            if not conservative:
                values["gamma"] = self.gamma * weight
            pair.params[(first, second)] = values
        return pair
