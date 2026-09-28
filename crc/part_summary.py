"""Headline CRC / Xenium numbers -> results/rerun_final/crc/part_summary_crc.csv."""
from pathlib import Path
import pandas as pd

R = Path("/Users/apple/Research/FlashDeconv/results/rerun_final/crc")
c = pd.read_csv(R / "claims_final.csv", index_col=0)
t = pd.read_csv(R / "timing_final.csv")
v020_t = pd.read_csv("/Users/apple/Research/FlashDeconv/results/rerun_v020/crc/timing_v020.csv")
v020_t = v020_t[v020_t.rep == 0]


def row(bench, metric, claim, man, notes=""):
    return dict(benchmark=bench, metric=metric, manuscript_value=man, v020_value=c.loc[claim, "V020"],
                final_value=c.loc[claim, "FINAL"], rank="", p_value="", notes=notes)


rows = [
    dict(benchmark="CRC Visium HD runtime", metric="fit_transform s total P1+P2+P5 (1,595,565 bins; 32 CPUs, sequential)",
         manuscript_value="153", v020_value=f"{v020_t.fit_seconds.sum():.1f}", final_value=f"{t.fit_seconds.sum():.1f}",
         rank="", p_value="", notes=f"FINAL iterations {'/'.join(map(str, t.n_iterations))} (converged all); "
         f"V020 stopped at max_iter=100 unconverged; peak RSS {t.peak_rss_gb.min():.1f}-{t.peak_rss_gb.max():.1f} GB; host c01 AMD EPYC 7763"),
    dict(benchmark="CRC Visium HD runtime", metric="bins per second", manuscript_value="10,400",
         v020_value=f"{v020_t.n_bins.sum() / v020_t.fit_seconds.sum():.0f}",
         final_value=f"{t.n_bins.sum() / t.fit_seconds.sum():.0f}", rank="", p_value="", notes=""),
    row("CRC neutrophil hotspots", "hotspot bins total (Neutrophil >= 0.10)", "hotspot bins total (16,827)", "16,827"),
    row("CRC neutrophil hotspots", "RCTD withheld % (reject+NA)", "RCTD withheld % (reject+NA; 61)", "61"),
    row("CRC neutrophil hotspots", "RCTD Neutrophil singlet %", "RCTD Neutrophil singlet % (2.3)", "2.3"),
    row("CRC neutrophil hotspots", "Neutrophil self-enrichment x P1/P2/P5", "Neutrophil self-enrichment x (16.6/22.8/56.2)", "16.6/22.8/56.2"),
    row("CRC neutrophil hotspots", "neutrophil marker FC range", "neutrophil marker FC range (11-63)", "11-63"),
    row("CRC neutrophil aggregates", "aggregates total", "aggregates total (72)", "72"),
    row("CRC neutrophil aggregates", "stromal-resident / tumor-proximal", "stromal-resident / tumor-proximal (25/47)", "25/47"),
    dict(benchmark="CRC neutrophil aggregates", metric="MWU SR>TP mRegDC", manuscript_value="",
         v020_value="", final_value="", rank="", p_value=f"ORIG {c.loc['MWU SR>TP mRegDC p', 'ORIG']}; V020 {c.loc['MWU SR>TP mRegDC p', 'V020']}; FINAL {c.loc['MWU SR>TP mRegDC p', 'FINAL']}", notes=""),
    row("CRC neutrophil neighborhood", "LAMP3 neighborhood fold median (per patient)", "LAMP3 neighborhood fold (1.40; 1.13-1.67)", "1.40 (1.13-1.67)",
        f"MWU p per patient ORIG {c.loc['LAMP3 MWU p per patient', 'ORIG']}; V020 {c.loc['LAMP3 MWU p per patient', 'V020']}; FINAL {c.loc['LAMP3 MWU p per patient', 'FINAL']} (P1 p unstable: 10k random subsample of neighborhood bins)"),
    row("CRC lineage disputes", "marker verdicts FD correct", "marker verdicts FD correct (19/22)", "19/22"),
    row("CRC tumor boundary", "tumor % at -25 / +75 um", "tumor % at -25um / +75um (94/14)", "94/14"),
    row("CRC multi-resolution", "Neut->mRegDC enrichment 8/16/32/64 um", "multires Neut-> mRegDC x (8/16/32/64um)", "",
        "ORIG used ~36/72/143 um bins (aggregation bug); V020/FINAL array-grid 16/32/64 um = Space Ranger square_016um"),
    row("CRC multi-resolution", "n_bins 8/16/32/64 um", "multires n_bins (8/16/32/64um)", "",
        "corrected counts scale 0.251-0.255 per doubling of bin side"),
]
x = pd.read_csv(R / "xenium" / "part_summary_xenium.csv")
out = pd.concat([pd.DataFrame(rows), x], ignore_index=True)
out.to_csv(R / "part_summary_crc.csv", index=False)
print(out.to_string())
