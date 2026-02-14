# MDR-TB CRISPR Assay – Design Improvement Plan (v4)

This note refines the design strategy used by `09_mdrtb_assay_design.py`.
It focuses on improving coverage of WHO-listed MDR-TB mutations in GC-rich
regions (especially rpoB RRDR) where classical Cas12a PAMs (TTTV) are sparse.

---

## A. Use more permissive CRISPR enzymes (future work)

- Classical Lb/AsCas12a requires a TTTV PAM, which is uncommon in the
  GC-rich rpoB Rifampicin Resistance Determining Region, so several important
  SNPs have no usable PAM within the current ±80 bp window.[file:108]
- Next-generation panels should consider engineered Cas12a variants with
  relaxed PAMs (e.g. TTYN / VTTV) or alternative nucleases (Cas12b, Cas13),
  but the current v3 script remains TTTV-based until an enzyme is selected.

---

## B. PAM-introducing RPA primers

For PAM-poor regions, especially rpoB:

- Allow a design mode where RPA primers *silently introduce* an optimal PAM
  (e.g. TTTV/TTYN) into the amplicon in the vicinity of the SNP.
- Only non-critical flanking bases are modified so that:
  - The SNP nucleotide itself remains unchanged.
  - The synthetic-mismatch (SM) discrimination strategy is preserved
    (1 mismatch vs mutant, 2 mismatches vs wild-type).
- This “primer-encoded PAM” approach is standard in SNP genotyping and can
  recover otherwise inaccessible targets while maintaining specificity.

Planned implementation:

- Add an optional flag (e.g. `--allow-primer-pam`) and, when enabled, extend
  primer search to include sequences that create a valid PAM in the amplicon
  while still passing primer quality filters (length, GC, homopolymer limits).

---

## C. Relax PAM window and focus panel scope

- Increase PAM search window for rpoB from ±80 bp to at least ±120–150 bp
  around each SNP so longer amplicons (≈220–260 bp) can be considered, which
  remain compatible with RPA-based assays.[file:108]
- Keep default amplicon sizes conservative (e.g. 120–200 bp) for katG and
  inhA promoter, but document that rpoB-specific profiles may use a slightly
  larger range after empirical validation.
- Prioritize SNPs that are both:
  - Clinically frequent and strongly associated with resistance.
  - Accessible either via native TTTV PAMs or primer-introduced PAMs.
- Aim first for a robust, buildable 6–8 target MDR-TB panel rather than a
  nominal 9–14 target panel with major PAM gaps.

Planned implementation:

- Expose the PAM window as a command-line option (e.g. `--pam-window-bp`)
  rather than a hard-coded 80 bp.
- Optionally support per-gene default windows (larger for rpoB RRDR).

---

## D. Soften in silico crRNA filters for high-GC M. tuberculosis

Given that the M. tuberculosis genome is ≈65–66 % GC, some current filters in
the v3 code are intentionally conservative and can be relaxed:

- **Spacer GC content**
  - Prefer 35–65 % GC but accept up to ~75 % GC with graded penalties instead
    of hard rejection, provided homopolymers and predicted secondary structure
    remain acceptable.

- **Spacer length**
  - Continue to support 17, 18, 20, and 23 nt spacers.
  - Score 17–18 nt slightly higher for discrimination, but keep 20–23 nt
    candidates in the pool for PAM-poor regions.

- **SNP position within spacer**
  - Prefer SNPs in positions 1–8 from the PAM (seed), but accept positions
    up to 9–10 when necessary and rank candidates using the existing scoring
    function rather than discarding them outright.

Planned implementation:

- In `score_crrna_design`, replace strict cutoffs with softer penalties so that
  suboptimal but still viable designs are retained and simply ranked lower.
- Log the number of candidates discarded by each filter (GC, seed position,
  spacer length) to make design bottlenecks explicit.

---

## Summary

The current v3 pipeline already surfaces an important limitation:
classical Cas12a with TTTV PAMs and a narrow ±80 bp window leaves several
WHO MDR-TB SNPs without accessible PAMs. The v4 design plan above documents
these constraints and outlines how to:

1. Introduce primer-level flexibility (PAM-introducing RPA primers).
2. Relax PAM search windows and in silico filters in a controlled way.
3. Focus the first panel on a smaller, experimentally realistic set of SNPs.
4. Prepare the codebase for future enzyme variants and learned guide scoring.

Code changes should be implemented incrementally and benchmarked against the
existing v3 behavior to ensure backward compatibility.
