from biocomp.network import Network, CoTransfection, TranscriptionUnit, Slot, Unit
import itertools as it

## {{{                      --     net generation     --
P = "hEF1a"
T = "L0.T_4560"
ERNS = ['CasE', 'Csy4', 'PgU']

UORFS = [
    None,
    "1w_uORF",
    "1x_uORF",
    "2x_uORF",
    "3x_uORF",
    "4x_uORF",
    "5x_uORF",
    "6x_uORF",
    "8x_uORF",
]

NO_UORFS = [None]


COLORS = {
    'x1': 'mKO2',
    'x2': 'eBFP2',
    'b': 'mMaroon1',
    'y': 'mNeonGreen',
}

u1 = Slot(
    part=UORFS,
    ref_id="U1",
)
u2 = Slot(
    part=UORFS,
    ref_id="U2",
)
u3 = Slot(
    part=UORFS,
    ref_id="U3",
)
u3 = Slot(
    part=NO_UORFS,
    ref_id="U3",
)


def make_units(tu_name, erns=None, mask=None):
    erns = erns or ERNS
    recs = [f"{ern}_rec" for ern in erns]
    N_MAX_UNITS = 8
    if mask is None:
        mask = [True] * N_MAX_UNITS

    assert len(mask) == N_MAX_UNITS, f"Mask length must be {N_MAX_UNITS}, got {len(mask)}"

    unmasked = [
        # marker
        Unit(slots=[P, COLORS[tu_name], T], name=f"{tu_name}_marker"),
        # a
        Unit(slots=[P, u1, recs[0], erns[2], T], name=f"{tu_name}_a+"),
        Unit(slots=[P, erns[0], T], name=f"{tu_name}_a-"),
        # b
        Unit(slots=[P, u2, recs[1], erns[2], T], name=f"{tu_name}_b+"),
        Unit(slots=[P, erns[1], T], name=f"{tu_name}_b-"),
        # c
        Unit(slots=[P, u3, recs[2], COLORS['y'], T], name=f"{tu_name}_c+"),
        Unit(slots=[P, erns[2], T], name=f"{tu_name}_c-"),
        # direct_out
        Unit(slots=[P, COLORS['y'], T], name=f"{tu_name}_direct_out"),
    ]
    units = [unmasked[i] for i in range(N_MAX_UNITS) if mask[i]]
    return units


def make_twoandone_network(erns=None):
    ern_names = ', '.join(erns) if erns else ', '.join(ERNS)
    return Network(
        name=f"two_and_one ({ern_names})",
        cotx=[
            CoTransfection(
                name="x1",
                units=make_units("x1", erns=erns),
            ),
            CoTransfection(
                name="x2",
                units=make_units("x2", erns=erns),
            ),
            CoTransfection(
                name="b",
                units=make_units("b", erns=erns),
            ),
        ],
        invert_on_build=True,
    )


def make_units_withskip(tu_name, erns=None, direct=True):
    erns = erns or ERNS
    recs = [f"{ern}_rec" for ern in erns]
    units = [
        # marker
        Unit(slots=[P, COLORS[tu_name], T], name=f"{tu_name}_marker"),
        # a
        Unit(slots=[P, u1, recs[0], erns[1], T], name=f"{tu_name}_a+"),
        Unit(slots=[P, erns[0], T], name=f"{tu_name}_a-"),
        # b
        Unit(slots=[P, u3, recs[1], COLORS['y'], T], name=f"{tu_name}_b+"),
        Unit(slots=[P, erns[1], T], name=f"{tu_name}_b-"),
        # c
        Unit(slots=[P, u3, recs[2], COLORS['y'], T], name=f"{tu_name}_c+"),
        Unit(slots=[P, erns[2], T], name=f"{tu_name}_c-"),
        # direct_out
    ]
    if direct:
        units.append(Unit(slots=[P, COLORS['y'], T], name=f"{tu_name}_direct_out"))
    return units


def make_twoandoneskip_network(erns=None):
    ern_names = ', '.join(erns) if erns else ', '.join(ERNS)
    return Network(
        name=f"two_and_one_skip ({ern_names})",
        cotx=[
            CoTransfection(
                name="x1",
                units=make_units_withskip("x1", erns=erns),
            ),
            CoTransfection(
                name="x2",
                units=make_units_withskip("x2", erns=erns),
            ),
            CoTransfection(
                name="b",
                units=make_units_withskip("b", erns=erns),
            ),
        ],
        invert_on_build=True,
    )


def make_units_three(tu_name, erns=None, direct=True):
    erns = erns or ERNS
    recs = [f"{ern}_rec" for ern in erns]
    units = [
        # marker
        Unit(slots=[P, COLORS[tu_name], T], name=f"{tu_name}_marker"),
        # a
        Unit(slots=[P, u1, recs[0], COLORS['y'], T], name=f"{tu_name}_a+"),
        Unit(slots=[P, erns[0], T], name=f"{tu_name}_a-"),
        # b
        Unit(slots=[P, u2, recs[1], COLORS['y'], T], name=f"{tu_name}_b+"),
        Unit(slots=[P, erns[1], T], name=f"{tu_name}_b-"),
        # c
        Unit(slots=[P, u3, recs[2], COLORS['y'], T], name=f"{tu_name}_c+"),
        Unit(slots=[P, erns[2], T], name=f"{tu_name}_c-"),
    ]
    if direct:
        units.append(Unit(slots=[P, COLORS['y'], T], name=f"{tu_name}_direct_out"))
    return units


def make_units_two(tu_name, erns=None, direct=True):
    erns = erns or ERNS
    recs = [f"{ern}_rec" for ern in erns]
    units = [
        # marker
        Unit(slots=[P, COLORS[tu_name], T], name=f"{tu_name}_marker"),
        # a
        Unit(slots=[P, u1, recs[0], COLORS['y'], T], name=f"{tu_name}_a+"),
        Unit(slots=[P, erns[0], T], name=f"{tu_name}_a-"),
        # b
        Unit(slots=[P, u2, recs[1], COLORS['y'], T], name=f"{tu_name}_b+"),
        Unit(slots=[P, erns[1], T], name=f"{tu_name}_b-"),
    ]
    if direct:
        units.append(Unit(slots=[P, COLORS['y'], T], name=f"{tu_name}_direct_out"))
    return units


def make_three_network(erns=None):
    ern_names = ', '.join(erns) if erns else ', '.join(ERNS)
    return Network(
        name=f"three ({ern_names})",
        cotx=[
            CoTransfection(
                name="x1",
                units=make_units_three("x1", erns=erns),
            ),
            CoTransfection(
                name="x2",
                units=make_units_three("x2", erns=erns),
            ),
            CoTransfection(
                name="b",
                units=make_units_three("b", erns=erns),
            ),
        ],
        invert_on_build=True,
    )


def make_two_network(erns=None):
    ern_names = ', '.join(erns) if erns else ', '.join(ERNS)
    return Network(
        name=f"two ({ern_names})",
        cotx=[
            CoTransfection(
                name="x1",
                units=make_units_two("x1", erns=erns),
            ),
            CoTransfection(
                name="x2",
                units=make_units_two("x2", erns=erns),
            ),
        ],
        invert_on_build=True,
    )


def make_all_networks():
    """
    Generate all networks with the given ERNs.
    """
    networks = []
    rotations = [ERNS[i:] + ERNS[:i] for i in range(len(ERNS))]
    permutations = list(it.permutations(ERNS))
    networks += [make_twoandone_network(erns=rot) for rot in rotations]
    networks += [make_twoandoneskip_network(erns=per) for per in permutations]
    networks += [make_three_network(erns=ERNS)]
    # for net in networks:
    #     net.set_input_as_bias('mMaroon1')
    networks += [make_two_network(erns=rot) for rot in rotations]
    return networks


ALL_NETWORKS = make_all_networks()
THREE_NETWORKS = [n for n in ALL_NETWORKS if n.name.startswith("three")]
TWO_AND_ONE_SKIP_NETWORKS = [n for n in ALL_NETWORKS if n.name.startswith("two_and_one_skip")]
TWO_AND_ONE_NETWORKS = [
    n
    for n in ALL_NETWORKS
    if n.name.startswith("two_and_one") and not n.name.startswith("two_and_one_skip")
]


##────────────────────────────────────────────────────────────────────────────}}}
