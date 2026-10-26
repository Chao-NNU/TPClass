# Reference data

This directory holds the reference data of TPClass.  The files are not tracked in git; run ash models/download.sh to fetch them, or drop your own copies in place.

## Files

### `tcdb_annot.clean.tsv`

TCDB (Transporter Classification Database) annotation table.

Format (TSV, header):
```
tc_number<TAB>description
1.A.1.1.1<TAB>Type A anion channel 1 (AarA)
1.A.1.1.2<TAB>Type A anion channel 2
...
```

Size: ~30,000 rows covering all TC families in TCDB.

Source: Saier laboratory, http://tcdb.org/

This file is **small (~5 MB)** and is downloaded by `bash models/download.sh` (it is not tracked in git).

### `refs/`

Reference FASTA files for substrate-specific transporter mining.

| File | Substrate | # Refs | Source |
|------|-----------|--------|--------|
| `hmo_refs_25.fasta` | HMO | 25 | 10 literature + 15 user-curated |
| `squalene_refs.fasta` | Squalene | 6 | Bacterial HpnN + POPT + CoxX |
| `isobutanol_refs.fasta` | Isobutanol | 18 | Yeast DHA1 + bacterial MFS + SMR |

#### How to add a new substrate

1. Curate 5-25 known substrate-related proteins from literature
2. Add their UniProt accessions to a new FASTA file in `refs/`
3. Format: `>sp|ACC|GENE_NAME description` (UniProt Swiss-Prot format) or `>ACC description`
4. Update the accessions list in `examples/<substrate>/README.md`

## Reference set curation guidelines

When curating a reference set:

1. **Include proteins with diverse TC family backgrounds** if the substrate has multiple TC families
2. **Prefer proteins with experimental validation** (uptake assays, knockouts, etc.)
3. **Include both model and non-model organisms** to capture cross-species variation
4. **Avoid metabolic enzymes** (e.g., ERG1, FPPS) which can cause V-ATPase c-subunit false positives due to ESM-2 structural similarity
5. **Size**: 5-25 proteins is optimal. Too few (<5) gives weak ESM-2 reference; too many (>30) dilutes the signal

## Sanity check

After running TPClass with a new reference set, verify that the **known** substrate-related proteins are in the top candidates. For example:
- AQR1 (P53943) for isobutanol
- YSP2 (A0ABX6F7C0) for squalene
- lacY (P02920) for HMO

If a known protein is not in the top, the reference set may need adjustment (e.g., add more diverse proteins, or change the filter thresholds).
