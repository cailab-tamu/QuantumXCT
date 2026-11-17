# LR_databases

Known human ligand-receptor (LR) pairs, used **only to annotate** QuantumXCT networks
(a discovered cross-cell edge that matches a known pair is drawn as a validated LR link).
QuantumXCT itself does not need a database: it can evaluate any cross-cell gene pair.

| File | Contents |
|---|---|
| `Human-merged-Jin-Dimitrov-LR-pairs.csv` | Union of the two resources below (`ligand, receptor, source`). This is the file the code reads. |
| `Human-2020-Jin-LR-pairs.csv` | CellChatDB human resource (Jin et al. 2021), kept for provenance. |
| `Human-2022-Dimitrov-LR-pairs.csv` | LIANA consensus resource (Dimitrov et al. 2022), kept for provenance. |
| `lr_pairs.py` | `curate_lr_pairs(genes)`: the known pairs among your genes. |

```python
from LR_databases.lr_pairs import curate_lr_pairs
curate_lr_pairs(["PDGFB", "STAT3", "HES4", "TGFB1", "PDGFRB", "TGFBR1", "TGFBR2"])
# [('PDGFB', 'PDGFRB'), ('TGFB1', 'TGFBR1'), ('TGFB1', 'TGFBR2')]
```

## Please cite the source databases

If you use these tables, cite both:

- Jin S, Guerrero-Juarez CF, Zhang L, *et al.* Inference and analysis of cell-cell
  communication using CellChat. *Nat Commun* **12**, 1088 (2021).
  https://doi.org/10.1038/s41467-021-21246-9
- Dimitrov D, Türei D, Garrido-Rodriguez M, *et al.* Comparison of methods and resources
  for cell-cell communication inference from single-cell RNA-Seq data.
  *Nat Commun* **13**, 3224 (2022). https://doi.org/10.1038/s41467-022-30755-0
