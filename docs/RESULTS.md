# RESULTS — paper vs bản tái hiện

> Quy tắc: mỗi số "Ours" phải có `run_id` (thư mục `runs/<run_id>/` có `metrics.json`). Stage 2 là mean ± std của 5 seed (42–46).
> Δ% = (Ours − Paper) / Paper × 100. Mục tiêu: |Δ| ≤ 5% (Beauty, ML-1M), ≤ 10% (Toys, Sports, Clothing).
> Số paper đầy đủ: `docs/PAPER_NOTES.md` §6–7.

## 1. Dữ liệu

### Checkpoint A (5-core + join meta, chưa cắt chuỗi) — so với bảng thống kê của paper
| Dataset | Users (paper / ours) | Items (paper / ours) | Interactions (paper / ours) | Avg len (paper / ours) | Hash |
|---|---|---|---|---|---|
| Toys | 19,406 / — | 11,842 / — | 166,230 / — | 8.6 / — | — |
| Beauty | 22,363 / — | 12,077 / — | 198,064 / — | 8.9 / — | — |
| Sports | 35,597 / — | 18,168 / — | 294,045 / — | 8.3 / — | — |
| Clothing | 39,387 / — | 22,445 / — | 270,319 / — | 6.9 / — | — |
| ML-1M | 6,040 / — | 3,397 / — | 996,867 / — | 165 / — | — |

### Chế độ `tail10` (code chính thức) — thống kê sau khi cắt
| Dataset | Users | Items | Interactions | Avg len | Ảnh lỗi bị loại | Hash |
|---|---|---|---|---|---|---|
| Toys | — | — | — | — | — | — |
| Beauty | — | — | — | — | — | — |
| Sports | — | — | — | — | — | — |
| Clothing | — | — | — | — | — | — |
| ML-1M | — | — | — | — | — | — |

## 2. SASRec ID-only (chẩn đoán D-01), N@20
| Dataset | Paper | Ours `full` | Δ% | Ours `tail10` | Δ% | run_id |
|---|---|---|---|---|---|---|
| Toys | 0.0602 | — | — | — | — | — |
| Beauty | 0.0483 | — | — | — | — | — |
| Clothing | 0.0260 | — | — | — | — | — |
| Sports | 0.0294 | — | — | — | — | — |
| ML-1M | 0.0773 | — | — | — | — | — |

## 3. VLM2Rec setting (không backbone), N@20
| Method | Toys (P/O) | Beauty (P/O) | Clothing (P/O) | Sports (P/O) | ML-1M (P/O) | run_id |
|---|---|---|---|---|---|---|
| VLM_Vanilla | 0.0570 / — | 0.0442 / — | 0.0245 / — | 0.0271 / — | 0.0711 / — | — |
| VLM_SFT | 0.0732 / — | 0.0695 / — | 0.0325 / — | 0.0358 / — | 0.1087 / — | — |
| **VLM2Rec** | 0.0823 / — | 0.0784 / — | 0.0370 / — | 0.0396 / — | 0.1252 / — | — |
| LLM_Vanilla | 0.0516 / — | 0.0422 / — | 0.0222 / — | 0.0244 / — | 0.0699 / — | — |
| LLM_SFT | 0.0693 / — | 0.0631 / — | 0.0306 / — | 0.0327 / — | 0.0817 / — | — |

## 4. VLM2Rec_plug setting (SASRec), N@20
| Method | Toys (P/O) | Beauty (P/O) | Clothing (P/O) | Sports (P/O) | ML-1M (P/O) | run_id |
|---|---|---|---|---|---|---|
| VLM_Vanilla | 0.0577 / — | 0.0463 / — | 0.0240 / — | 0.0262 / — | 0.0695 / — | — |
| VLM_SFT | 0.0699 / — | 0.0622 / — | 0.0325 / — | 0.0326 / — | 0.1082 / — | — |
| **VLM2Rec_plug** | 0.0786 / — | 0.0728 / — | 0.0345 / — | 0.0378 / — | 0.1216 / — | — |
| LLM_SFT | 0.0715 / — | 0.0612 / — | 0.0310 / — | 0.0325 / — | 0.1052 / — | — |

## 5. Dropout test (VLM2Rec setting, N@20): TEXT / VISION / FUSION (Gap%)
| Data | Model | Paper | Ours | run_id |
|---|---|---|---|---|
| Beauty | VLM_Vanilla | .045 / .042 / .0442 (6.9%) | — | — |
| Beauty | VLM_SFT | .072 / .0645 / .0695 (12.1%) | — | — |
| Beauty | VLM2Rec | .078 / .077 / .0784 (1.3%) | — | — |
| ML-1M | VLM_Vanilla | .070 / .065 / .0711 (7.2%) | — | — |
| ML-1M | VLM_SFT | .114 / .0995 / .1087 (14.6%) | — | — |
| ML-1M | VLM2Rec | .1225 / .120 / .1252 (2.0%) | — | — |

## 6. Kiểm chứng quyết định (variant)
| Quyết định | Dataset | Phương án A → N@20 | Phương án B → N@20 | Kết luận | run_id |
|---|---|---|---|---|---|
| D-01 | — | — | — | — | — |

## 7. Nhật ký run
| run_id | Stage | Dataset | Method | Variant/flags | Thiết bị | Kết quả chính | Ghi chú |
|---|---|---|---|---|---|---|---|
