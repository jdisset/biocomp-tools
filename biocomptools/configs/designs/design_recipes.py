from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit, Slot, NumRange, FluoIntensity
from biocomp.network import recipe_to_networks

P = "hEF1a"
T = "L0.T_4560"
ERNS = ["CasE", "Csy4", "PgU"]
UORFS = [None, "1w_uORF", "1x_uORF", "2x_uORF", "3x_uORF", "4x_uORF", "5x_uORF", "6x_uORF", "8x_uORF"]
COLORS = {"x1": "mKO2", "x2": "eBFP2", "b": "mMaroon1", "y": "mNeonGreen"}
BIAS_FLUO = FluoIntensity(tu_id=0, value=NumRange(min=0.3, max=0.6), protein=COLORS["b"], units="Rescaled AU")


def make_design_units(tu_name, erns=None):
    erns = erns or ERNS
    recs = [f"{ern}_rec" for ern in erns]
    u1 = Slot(part=UORFS, ref_id="U1")
    u2 = Slot(part=UORFS, ref_id="U2")
    u3 = Slot(part=UORFS, ref_id="U3")
    return [
        TranscriptionUnit(slots=[P, COLORS[tu_name], T], name=f"{tu_name}_marker"),
        TranscriptionUnit(slots=[P, u1, recs[0], erns[2], T], name=f"{tu_name}_a+"),
        TranscriptionUnit(slots=[P, erns[0], T], name=f"{tu_name}_a-"),
        TranscriptionUnit(slots=[P, u2, recs[1], erns[2], T], name=f"{tu_name}_b+"),
        TranscriptionUnit(slots=[P, erns[1], T], name=f"{tu_name}_b-"),
        TranscriptionUnit(slots=[P, u3, recs[2], COLORS["y"], T], name=f"{tu_name}_c+"),
        TranscriptionUnit(slots=[P, erns[2], T], name=f"{tu_name}_c-"),
        TranscriptionUnit(slots=[P, COLORS["y"], T], name=f"{tu_name}_direct_out"),
    ]


def make_twoandone_design_recipe(erns=None):
    erns = erns or ERNS
    ern_names = ", ".join(erns)
    recipe = Recipe(
        name=f"two_and_one_design ({ern_names})",
        content=[
            CoTransfection(
                name="x1",
                units=make_design_units("x1", erns=erns),
                ratios=[NumRange(min=0.5, max=10.0) for _ in range(8)],
            ),
            CoTransfection(
                name="x2",
                units=make_design_units("x2", erns=erns),
                ratios=[NumRange(min=0.5, max=10.0) for _ in range(8)],
            ),
            CoTransfection(
                name="b",
                units=make_design_units("b", erns=erns),
                ratios=[NumRange(min=0.5, max=10.0) for _ in range(8)],
                fluo_bias=BIAS_FLUO,
            ),
        ],
    )
    networks = recipe_to_networks(recipe, invert=True, inversion_mode="main")
    return networks[0] if networks else None


DESIGN_TWOANDONE = make_twoandone_design_recipe()
