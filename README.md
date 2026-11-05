# DCASE 2026 Challenge Task 7 — 2nd Place (Team Ranking)

**Domain-Incremental Sound Event Classification** · Team Chun_Chosun (Chosun University)

- 🥈 **2nd place** in the official team ranking
- 🥇 System 1: **highest D1 accuracy (91%) among all participating teams** — best retention of the initial domain
- Method: LoRA adapters on a frozen backbone + Orthogonal Gradient Projection + Knowledge Distillation (routing-free, rehearsal-free)
- Follow-up paper: *OR-KDL: Orthogonal Knowledge-Distilled LoRA for Domain-Agnostic Incremental Learning* (submitted to ICASSP 2027)

---

# Chun_Chosun_task7 — DCASE 2026 Challenge Task 7 Submission

This package contains the submission files for **Task 7** by Chun (Chosun University),
covering four systems (`Chun_Chosun_task7_1` … `Chun_Chosun_task7_4`).

## Model checkpoints (hosted externally)

The trained model weights (D1 / D2 / D3 checkpoints) are **not included** in this
archive because of their size (~40 GB total). They are hosted on Google Drive:

**https://drive.google.com/drive/folders/1RZOGw4nUFIsj8NDBRuKDV8oiI3SBQH4D**

The Drive folder contains one archive per system:

| File | Contents |
|------|----------|
| `Chun_Chosun_task7_1_checkpoints.tar.gz` | System 1 weights — D2 dictionary (5-fold) + D3 dictionary (25) |
| `Chun_Chosun_task7_2_checkpoints.tar.gz` | System 2 weights — D2 dictionary (5-fold) + D3 dictionary (25) |
| `Chun_Chosun_task7_3_checkpoints.tar.gz` | System 3 weights — D2 dictionary (5-fold) + D3 dictionary (25) |
| `Chun_Chosun_task7_4_checkpoints.tar.gz` | System 4 weights — D2 dictionary (5-fold) + D3 dictionary (25) |

Each archive holds **30 checkpoints** (`*_D2_dictionary/*.pth` and `*_D3_dictionary/*.pth`).
After download, extract with:

```bash
tar -xzf Chun_Chosun_task7_1_checkpoints.tar.gz -C Chun_Chosun_task7_1/
```

## Package layout

```
Chun_Chosun_task7/
├── README.md
├── Chun_Chosun_task7.technical_report.pdf   # technical report
├── Chun_Chosun_task7_1/                      # System 1
│   ├── Chun_Chosun_task7_1.meta.yaml         # meta information
│   ├── Chun_Chosun_task7_1.output.csv        # system output
│   └── Chun_Chosun_task7_1_model/            # model definition / scripts
├── Chun_Chosun_task7_2/                      # System 2
├── Chun_Chosun_task7_3/                      # System 3
└── Chun_Chosun_task7_4/                      # System 4
```

> Model weight files (`*.pth`) are excluded here — download them from the Drive link above.
