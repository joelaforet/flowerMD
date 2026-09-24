All-atom PhantomWalk initialization
===================================

The all-atom PhantomWalk workflow builds a dense amorphous polymer melt at
its target density and returns coordinates that a standard force field can
minimize. It does not produce an equilibrated melt; it produces a starting
structure without the overlaps that make direct minimization at density
fail.

The workflow has four steps, one flowerMD class each:

1. **Chains**: a `Polymer` preset such as ``PolyEthylene``, ``P3HT``,
   ``PES``, ``PolyStyrene``, ``PMMA``, ``PET``, ``Polycarbonate``, ``PEI``
   or ``PIM1``, or any ``MarkedSmilesPolymer`` subclass.
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
=================================  =========================================

Every value is an argument. Bonded terms can be switched off one kind at a
time (``include_bonds``, ``include_angles``, ``include_dihedrals``,
``include_impropers``) for ablations, the stopping rule can be replaced by
any ``stop(sim)`` callable, and ``flowermd.utils.schulz_zimm_lengths``
gives polydisperse ``lengths`` and ``num_mols``. A capped DPD run still
returns coordinates, with ``record["dpd_converged"] = False`` and a
warning; ``require_convergence=True`` makes it an error.

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
