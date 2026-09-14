These fixtures contain six undistorted H-capped donors and eighteen frozen
finite-chain chemical graphs. They support chemical assembly tests only.
No generated chain positions or dynamics are asserted.

Donors came from the frozen star_polymer._template function with the recorded
seed in the isolated construction Pixi environment. The JSON records source
hashes and package versions. Correspondence follows the frozen resolver's heavy
atom indices and the donor's ordered hydrogen pools. Symmetry-equivalent H
choices are recorded explicitly; mapping does not rely on unlabelled graph
isomorphism. Donor coordinates have angstrom units.

The frozen final graphs and stereo references come from the degree 1, 2 and 3
short construction fixtures. Every atom maps to a repeat and original donor
index. Original fixture hashes are retained. All mapped bond orders agree after
the same RDKit conversion and sanitization used by the frozen builder.
PIM-1 junction bonds enter as single bonds and become aromatic after
sanitization. Both orders are recorded separately. The test checks each directed
attachment pair before comparing its sanitized bond order.

Capture script: capture-chemical-donors-20260914.py in the local test-env evidence.
Capture script SHA256: 232506235dce15eaa945b01b1c873775bff2d152b9320b46613e6015edeb1318.
Frozen donor JSON SHA256: 415608970d0310644bb77e423ae25e1952b1c4d96fe2f93dec62880c71042139.
