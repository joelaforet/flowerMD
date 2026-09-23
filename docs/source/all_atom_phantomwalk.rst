All-atom PhantomWalk initialization
===================================

The all-atom PhantomWalk workflow builds a dense amorphous polymer melt at
its target density and returns coordinates that a standard force field can
minimize. It does not produce an equilibrated melt; it produces a starting
structure without the overlaps that make direct minimization at density
fail.

The workflow has four steps, one flowerMD class each:

1. **Chains**: a `Polymer` preset such as ``PolyEthylene``, ``P3HT``,
   ``PES``, ``PolyStyrene``, ``PMMA``, ``PET``, ``Polycarbonate``, ``PEI``,
   ``PIM1`` or the ionomer ``PEAAIonomer``, or any ``MarkedSmilesPolymer``
   subclass.
2. **Placement at the target density**: ``AllAtomRandomWalk`` turns the
   bonds between repeat units to random torsions, so chains start as random
   coils; ``AllAtomLattice`` places whole chains in their built
   conformation. Both keep every bond length, bond angle and stereocenter
   of the built chains. Chains overlap freely; that is intended.
3. **Interactions**: ``AllAtomDPD``, bonded terms from UFF (default) or
   Sage 2.3.0, scaled up, with a soft DPD pair force whose coefficients are
   weighted by each pair's Lennard-Jones well depth. Stereocenters are
   recorded and protected automatically.
4. **Relaxation**: ``AllAtomPhantomWalk.run_initialization`` runs DPD until
   every force's per-particle energy is stationary, then FIRE with
   conservative DPD repulsion, and returns a run record.

.. code-block:: python

    import unyt as u
    from flowermd.library import (
        AllAtomDPD, AllAtomPhantomWalk, AllAtomRandomWalk, PolyStyrene,
    )

    chains = PolyStyrene(lengths=100, num_mols=13, tacticity="atactic")
    system = AllAtomRandomWalk(chains, density=1.04 * u.g / u.cm**3)
    ff = AllAtomDPD(system.system)
    sim = AllAtomPhantomWalk.from_system(system, forcefield=ff)
    record = sim.run_initialization()
    compound = sim.to_compound()  # nm coordinates for OpenFF, foyer, ...

Default protocol
----------------

=================================  =========================================
Setting                            Default
=================================  =========================================
Bonded source                      UFF (``bonded="openff"`` for Sage 2.3.0)
Bonded scale                       30
DPD repulsion ``A`` / friction     1250 / 200, weighted by sqrt(eps_i eps_j)/eps_max
``kT``, cutoff, time step          1 kcal/mol, 3.5 Å, 0.001
Stopping rule                      five samples per 500-step chunk after a
                                   3500-step prelude; stop when every force
                                   changes by at most 2 % for two chunks in
                                   a row; cap 40,000 steps
FIRE                               100 steps, conservative DPD
Stereochemistry guard              on, k = 30,000 kcal/mol
Electrostatics                     off (``electrostatics="smeared"``, below)
=================================  =========================================

Every value is an argument. Bonded terms can be switched off one kind at a
time (``include_bonds``, ``include_angles``, ``include_dihedrals``,
``include_impropers``) for ablations, the stopping rule can be replaced by
any ``stop(sim)`` callable, and ``flowermd.utils.schulz_zimm_lengths``
gives polydisperse ``lengths`` and ``num_mols``. A capped DPD run still
returns coordinates, with ``record["dpd_converged"] = False`` and a
warning; ``require_convergence=True`` makes it an error.

Charged systems and smeared electrostatics
------------------------------------------

The default protocol has no electrostatics: point charges cannot be used
with a soft DPD core, because two opposite charges passing through each
other have an unbounded energy. For ionic polymers the optional smeared
term replaces each charge with a Gaussian cloud of standard deviation
``charge_smearing`` (default 2 Å). The pair energy becomes
``332.06 q_i q_j erf(r / (2 sigma)) / r`` kcal/mol: Coulomb beyond a few
``sigma``, and finite at contact. HOOMD's PPPM mesh computes it on the GPU,
with bonded 1-2, 1-3 and 1-4 pairs excluded as for the DPD pair.

.. code-block:: python

    from flowermd.library import PEAAIonomer

    chains = PEAAIonomer(lengths=7, num_mols=200, pattern="EEAEE")
    system = AllAtomRandomWalk(
        [chains, chains.counterions("[Na+]")], density=0.85 * u.g / u.cm**3
    )
    ff = AllAtomDPD(system.system, charges="nagl", electrostatics="smeared")

``charges`` is ``"formal"``, ``"gasteiger"``, ``"nagl"`` (the AM1-BCC-like
charges Sage 2.3.0 itself assigns) or an array with one value per particle;
the system must be neutral when ``electrostatics="smeared"``. Formal
charges (a carboxylate oxygen, a sodium ion) are inferred from each atom's
explicit valence, since mBuild does not store them.

Validation
----------

The workflow was checked against the reference PhantomWalk implementation
used for the all-atom PhantomWalk study, on about 20,000-atom melts with
Sage 2.3.0 minimization of the returned coordinates (OpenMM, CUDA). The
table gives the energy the minimizer removed per atom, in units of the
largest Sage Lennard-Jones well depth; lower means closer to
minimizer-ready.

=============  ====  =================  ===================
Chemistry      Runs  flowerMD (median)  Reference (median)
=============  ====  =================  ===================
PE             3     4.05               4.24
P3HT           3     4.26               4.22
PES            6     6.23               8.16
=============  ====  =================  ===================

Fresh atactic polystyrene and PMMA melts (three each) kept all 8,019
tetrahedral stereocenters through DPD, FIRE and unbiased Sage
minimization. PIM-1 (three melts, built with ``LadderPolymer``) reached
stationarity and minimized cleanly. No run reached the step cap.

See ``tutorials/7-flowerMD-all-atom-phantomwalk.ipynb`` for a complete
example that runs on a CPU in about 20 seconds.
