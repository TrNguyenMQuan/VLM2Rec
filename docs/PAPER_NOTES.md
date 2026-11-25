# PAPER_NOTES — VLM2Rec (arXiv:2603.17450v3)

> Distilled, citation-checked notes for implementation. Source of truth = `paper/arxiv_2603.17450v3/` (TeX v3, 2 Nov 2026).
> Where the official code disagrees, see `docs/REFERENCE_CODE_NOTES.md` and `docs/DECISIONS.md`.

## 0. Bibliographic
- Title: *VLM2Rec: Resolving Modality Collapse in Vision-Language Models for Sequential Recommendation*
- Authors: Junyoung Kim, Woojoo Kim, Jaehyung Lim, Dongha Kim (POSTECH), Wonbin Kweon* (SKKU), Hwanjo Yu* (POSTECH). WWW '27 (Dublin).
- Versions: v1 2026-03-18, v2 2026-08-12, **v3 2026-11-02 (used here)**. License CC BY-NC-ND 4.0.
- Code: commented-out link in abstract → `https://anonymous.4open.science/r/VLM2Rec/` (fetched via API, copy in `third_party/vlm2rec_anon/`).

## 1. Problem & embeddings (Sec. 3, "Modality Embeddings" in Sec. 4)
- Items $\mathcal I$, user history $S_u=[i_1..i_L]$, each item has **title + image**. Predict next item $i^+$.
- VLM $\Phi$ = Qwen2.5-VL-3B(-Instruct). Embedding = **ℓ2-normalized last-token hidden state of the final layer**.
- **Separate prompts per modality** (each reads the whole sequence in one context):
  - $P_T$: `Describe this item text sequence in one word, 0) <title_1>, ..., L-1) <title_L>`
  - $P_V$: `Describe this item image sequence in one word, 0) <image_1>, ..., L-1) <image_L>`
  - Item embedding = same prompt with a single item (L=1).
  - (Official code strings differ slightly — "title" instead of "text", `., ` punctuation, `0). ` numbering. See REFERENCE_CODE_NOTES §3.)
- $\mathbf e_u^m=\Phi(P_m(S_u^m))$, $\mathbf e_i^m=\Phi(P_m(i^m))$, $m\in\{T,V\}$.
- **Fusion = sum, no extra params**: $\mathbf e_u=\mathbf e_u^T+\mathbf e_u^V$, $\mathbf e_i=\mathbf e_i^T+\mathbf e_i^V$.
- Ranking: all items by $s(\mathbf e_u,\mathbf e_i)$ (paper: cosine; code: dot product of the un-normalized sums).

## 2. Losses
Notation: batch $\mathcal B$, $s$=cosine, $\mathcal N_u$ = **next items of the other users in the batch**, $\mathcal C$ = in-batch items.

**Standard contrastive SFT (Eq. 2)** — baseline $\text{VLM}_\text{SFT}$:
$$\mathcal L_\text{SFT}=-\mathbb E_{u\in\mathcal B}\log\frac{e^{s(\mathbf e_u,\mathbf e_{i^+})/\tau}}{e^{s(\mathbf e_u,\mathbf e_{i^+})/\tau}+\sum_{i^-\in\mathcal N_u}e^{s(\mathbf e_u,\mathbf e_{i^-})/\tau}}$$

**WPCL (Sec. 5.1)**
- Discriminative margin per user & modality:
  $\mathcal M_{u,m}=s(\mathbf e_u^m,\mathbf e^m_{i^+})-\frac1{|\mathcal N_u|}\sum_{i^-}s(\mathbf e_u^m,\mathbf e^m_{i^-})$
- $\mathcal M_{u,\text{strong}}=\max_m\mathcal M_{u,m}$, $\mathcal M_{u,\text{weak}}=\min_m\mathcal M_{u,m}$ (re-identified every step, per user).
- Gap: $\Delta_{u,\text{gap}}=\text{sg}[\mathcal M_{u,\text{strong}}]-\mathcal M_{u,\text{weak}}$ (stop-grad on strong only).
- Penalty: $w_{u,\text{pen}}=1+\text{Softplus}(\Delta_{u,\text{gap}})$ — **not detached** (gradient flows through $w$ into the weak margin).
- Loss: same as SFT on **fused** embeddings but the negative sum is multiplied by $w_{u,\text{pen}}$, temperature $\tau_\text{WPCL}$.

**CRTR (Sec. 5.2)**
- $\mathbf P^m_{u,j}=\text{softmax}_{j\in\mathcal C}\big(s(\mathbf e_u^m,\mathbf e_j^m)/\tau_\text{CRTR}\big)$
- $\mathcal L_\text{CRTR}=\tfrac12\mathbb E_u[\text{KL}(\mathbf P^T_u\|\mathbf P^V_u)+\text{KL}(\mathbf P^V_u\|\mathbf P^T_u)]$ (bidirectional, no fixed target).

**Final**: $\mathcal L=\mathcal L_\text{WPCL}+\lambda\,\mathcal L_\text{CRTR}$. Only LoRA params are updated.

**Algorithm 1**: per minibatch → encode histories & targets with $P_T,P_V$ → normalize last-token states, sum → margins/weak/strong/Δ/w → WPCL + CRTR → update LoRA. After training: encode each item once → fused table $\mathbf E$.

## 3. VLM2Rec_plug (Sec. 5.4)
- Frozen $\mathbf E$ (fused item embeddings) replaces the SR backbone's item table through a **linear adapter W**:
  $\hat y_{u,i}=f_\psi(\mathbf W\mathbf e_{i_1},\dots,\mathbf W\mathbf e_{i_L})^\top\mathbf W\mathbf e_i$. Train only $\mathbf W,\psi$ with the backbone's own loss.
- Hidden size d=128 (SASRec). No VLM call at serving time.

## 4. Analysis definitions (Sec. 4)
- **Dropout test** TEXT / VISION / FUSION: rank with $\mathbf e_u^T$ / $\mathbf e_u^V$ / $\mathbf e_u$ against the **fused** item embeddings. In the plug setting (macro comment in `paper_preamble.tex`): history items use text-only / vision-only / fused item embeddings, the next item is always fused.
- **Gap (%)** = (TEXT − VISION) / VISION × 100.
- **Gradient gap**: feed fused / text-only / vision-only sequence embeddings into $\mathcal L_\text{SFT}$, take grads w.r.t. each → $\mathbf g_\text{fusion},\mathbf g_T,\mathbf g_V$; track $\cos(\mathbf g_\text{fusion},\mathbf g_m)$ over training; gap = difference; reported as mean over last 20% of training (smoothed): SFT 0.74 (Beauty) / 0.87 (ML); VLM2Rec −0.27 / −0.08.
- **Geometry** (Wang & Isola): $U^m=\log\mathbb E_{(u,i)\sim\text{uniform}}e^{-2\|\mathbf e^m_u-\mathbf e^m_i\|^2}$; $A^m_\text{pos}=\mathbb E\|\mathbf e^m_u-\mathbf e^m_{i^+}\|^2$; $A^m_\text{neg}=\mathbb E\mathbb E_{i^-}\|\mathbf e^m_u-\mathbf e^m_{i^-}\|^2$; separability $S^m=A^m_\text{neg}/(A^m_\text{pos}+\epsilon)$.
- **Head-SFT control** (App.): freeze VLM, train only a linear head with the SFT loss → gap still widens.

## 5. Experimental setup (Sec. 6.1 + App. A)
- Datasets: Amazon 2014 (McAuley SIGIR'15) Toys, Beauty, Clothing, Sports + MovieLens-1M. 5-core, leave-one-out. Items lacking title or image removed. ML-1M posters from TMDB via the MovieLens links file (IMDb/TMDB IDs).
- Eval: **full ranking**; H@K, N@K, K∈{10,20}; mean of **5 seeds**; paired t-test p<0.05.
- VLM: Qwen2.5-VL-3B; LLM baselines: Qwen2.5-3B. LoRA r=16, α=32, dropout 0.2. **3 epochs, AdamW lr 1e-5, batch 32**, gradient checkpointing, one RTX 3090 (A100-80GB for larger models).
- Grid: τ_WPCL ∈ {0.05, 0.1, 0.5, 1.0}; τ_CRTR ∈ {0.001, 0.01, 0.1, 1.0}; λ ∈ {0.1, 0.5, 1.0}. (Selected values not reported; see §7.6.)
- $\mathcal N_u$ and $\mathcal C$ from in-batch items.
- SR backbones: Adam lr 1e-3 (β 0.9/0.999), batch 256, dropout 0.5, early stop patience 10 on valid N@20, d=128. One RTX 3090.
- Baseline details: BERT = bert-large-uncased (item-wise, sequence = mean-pool of item embs); CLIP = clip-vit-large-patch14 (same pooling); MLLM-MSR's yes/no loss replaced by SFT; SLIM teacher = gpt-4.1-mini.

### Dataset statistics (App. Tab. "dataset_stats") — reproduce at "Checkpoint A" (before truncation)
| Dataset | #Users | #Items | #Interactions | Avg len | Sparsity |
|---|---|---|---|---|---|
| Toys | 19,406 | 11,842 | 166,230 | 8.6 | 99.93% |
| Beauty | 22,363 | 12,077 | 198,064 | 8.9 | 99.93% |
| Sports | 35,597 | 18,168 | 294,045 | 8.3 | 99.95% |
| Clothing | 39,387 | 22,445 | 270,319 | 6.9 | 99.97% |
| MovieLens-1M | 6,040 | 3,397 | 996,867 | 165 | 95.14% |

Reference (raw 5-core, Amazon 2014): Beauty 22,363/12,101/198,502; Toys 19,412/11,924/167,597; Sports 35,598/18,357/296,337; Clothing 39,387/23,033/278,677; ML-1M 6,040/3,706/1,000,209. → The table equals "5-core then inner-join with metadata having title & imUrl", **without** re-applying 5-core and **without** sequence truncation.

## 6. Target numbers — main table (Tab. "perf_task2")
`*` = significant vs strongest baseline. Bold/underline from the paper dropped.

#### ID-based (shared)

| Method | Toys H@10 | Toys H@20 | Toys N@10 | Toys N@20 | Beauty H@10 | Beauty H@20 | Beauty N@10 | Beauty N@20 | Clothing H@10 | Clothing H@20 | Clothing N@10 | Clothing N@20 | Sports H@10 | Sports H@20 | Sports N@10 | Sports N@20 | MovieLens H@10 | MovieLens H@20 | MovieLens N@10 | MovieLens N@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU4Rec | 0.0662 | 0.0963 | 0.0362 | 0.0436 | 0.0638 | 0.0955 | 0.0350 | 0.0428 | 0.0312 | 0.0519 | 0.0153 | 0.0203 | 0.0326 | 0.0504 | 0.0171 | 0.0214 | 0.0962 | 0.1495 | 0.0471 | 0.0593 |
| FMLP | 0.0840 | 0.1138 | 0.0487 | 0.0561 | 0.0665 | 0.0987 | 0.0369 | 0.0451 | 0.0349 | 0.0564 | 0.0173 | 0.0224 | 0.0375 | 0.0564 | 0.0208 | 0.0255 | 0.1091 | 0.1602 | 0.0547 | 0.0673 |
| DiffuRec | 0.0890 | 0.1240 | 0.0500 | 0.0590 | 0.0851 | 0.1225 | 0.0471 | 0.0568 | 0.0488 | 0.0730 | 0.0252 | 0.0315 | 0.0473 | 0.0696 | 0.0250 | 0.0309 | 0.1296 | 0.1901 | 0.0643 | 0.0787 |
| SASRec | 0.0904 | 0.1221 | 0.0524 | 0.0602 | 0.0713 | 0.1053 | 0.0397 | 0.0483 | 0.0402 | 0.0648 | 0.0198 | 0.0260 | 0.0428 | 0.0648 | 0.0240 | 0.0294 | 0.1263 | 0.1830 | 0.0629 | 0.0773 |

#### VLM2Rec setting (no SR backbone)

| Method | Toys H@10 | Toys H@20 | Toys N@10 | Toys N@20 | Beauty H@10 | Beauty H@20 | Beauty N@10 | Beauty N@20 | Clothing H@10 | Clothing H@20 | Clothing N@10 | Clothing N@20 | Sports H@10 | Sports H@20 | Sports N@10 | Sports N@20 | MovieLens H@10 | MovieLens H@20 | MovieLens N@10 | MovieLens N@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BERT | 0.0550 | 0.0810 | 0.0310 | 0.0376 | 0.0433 | 0.0686 | 0.0223 | 0.0286 | 0.0234 | 0.0408 | 0.0113 | 0.0155 | 0.0244 | 0.0398 | 0.0119 | 0.0157 | 0.0679 | 0.1036 | 0.0320 | 0.0408 |
| SLIM | 0.0576 | 0.0850 | 0.0323 | 0.0390 | 0.0443 | 0.0708 | 0.0232 | 0.0297 | 0.0241 | 0.0426 | 0.0118 | 0.0162 | 0.0254 | 0.0412 | 0.0124 | 0.0164 | 0.0695 | 0.1066 | 0.0327 | 0.0420 |
| SLIM+ | 0.0971 | 0.1379 | 0.0547 | 0.0647 | 0.0795 | 0.1213 | 0.0439 | 0.0544 | 0.0430 | 0.0673 | 0.0215 | 0.0276 | 0.0419 | 0.0640 | 0.0220 | 0.0276 | 0.1270 | 0.1909 | 0.0618 | 0.0780 |
| LLM_Vanilla | 0.0781 | 0.1095 | 0.0436 | 0.0516 | 0.0593 | 0.0913 | 0.0342 | 0.0422 | 0.0332 | 0.0545 | 0.0169 | 0.0222 | 0.0351 | 0.0563 | 0.0192 | 0.0244 | 0.1133 | 0.1739 | 0.0547 | 0.0699 |
| LLM_SFT | 0.1041 | 0.1467 | 0.0586 | 0.0693 | 0.0952 | 0.1539 | 0.0485 | 0.0631 | 0.0489 | 0.0745 | 0.0242 | 0.0306 | 0.0511 | 0.0756 | 0.0266 | 0.0327 | 0.1331 | 0.1975 | 0.0656 | 0.0817 |
| VLM_Vanilla | 0.0843 | 0.1197 | 0.0481 | 0.0570 | 0.0674 | 0.1054 | 0.0351 | 0.0442 | 0.0384 | 0.0610 | 0.0189 | 0.0245 | 0.0415 | 0.0605 | 0.0224 | 0.0271 | 0.1188 | 0.1740 | 0.0570 | 0.0711 |
| VLM_SFT | 0.1132 | 0.1456 | 0.0652 | 0.0732 | 0.0942 | 0.1527 | 0.0547 | 0.0695 | 0.0471 | 0.0807 | 0.0241 | 0.0325 | 0.0506 | 0.0817 | 0.0279 | 0.0358 | 0.1671 | 0.2527 | 0.0869 | 0.1087 |
| MLLM-MSR | 0.1092 | 0.1504 | 0.0608 | 0.0711 | 0.0994 | 0.1573 | 0.0512 | 0.0655 | 0.0491 | 0.0750 | 0.0248 | 0.0313 | 0.0521 | 0.0788 | 0.0273 | 0.0342 | 0.1552 | 0.2259 | 0.0805 | 0.0982 |
| I-LLMRec | 0.1116 | 0.1549 | 0.0621 | 0.0730 | 0.1047 | 0.1629 | 0.0554 | 0.0700 | 0.0517 | 0.0803 | 0.0258 | 0.0331 | 0.0545 | 0.0808 | 0.0292 | 0.0357 | 0.1699 | 0.2474 | 0.0886 | 0.1085 |
| **VLM2Rec** | 0.1254* | 0.1682* | 0.0720* | 0.0823* | 0.1161* | 0.1732* | 0.0641* | 0.0784* | 0.0553* | 0.0875* | 0.0290* | 0.0370* | 0.0585* | 0.0904* | 0.0316* | 0.0396* | 0.1899* | 0.2783* | 0.1029* | 0.1252* |
| Imp.(%) | 10.78 | 8.59 | 10.43 | 12.43 | 10.89 | 6.32 | 15.70 | 12.00 | 6.96 | 8.43 | 12.40 | 11.78 | 7.34 | 10.65 | 8.22 | 10.61 | 11.77 | 10.13 | 16.14 | 15.18 |

#### VLM2Rec_plug setting (SASRec backbone)

| Method | Toys H@10 | Toys H@20 | Toys N@10 | Toys N@20 | Beauty H@10 | Beauty H@20 | Beauty N@10 | Beauty N@20 | Clothing H@10 | Clothing H@20 | Clothing N@10 | Clothing N@20 | Sports H@10 | Sports H@20 | Sports N@10 | Sports N@20 | MovieLens H@10 | MovieLens H@20 | MovieLens N@10 | MovieLens N@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BERT | 0.0702 | 0.1037 | 0.0387 | 0.0471 | 0.0601 | 0.0952 | 0.0308 | 0.0395 | 0.0303 | 0.0517 | 0.0144 | 0.0197 | 0.0314 | 0.0511 | 0.0157 | 0.0207 | 0.0884 | 0.1337 | 0.0397 | 0.0510 |
| SLIM | 0.0733 | 0.1077 | 0.0412 | 0.0497 | 0.0606 | 0.0957 | 0.0311 | 0.0397 | 0.0306 | 0.0525 | 0.0147 | 0.0201 | 0.0324 | 0.0519 | 0.0162 | 0.0210 | 0.0901 | 0.1347 | 0.0403 | 0.0512 |
| SLIM+ | 0.0961 | 0.1332 | 0.0561 | 0.0655 | 0.0812 | 0.1205 | 0.0443 | 0.0540 | 0.0434 | 0.0681 | 0.0218 | 0.0280 | 0.0427 | 0.0656 | 0.0229 | 0.0287 | 0.1555 | 0.2341 | 0.0779 | 0.0978 |
| LLMEmb | 0.0985 | 0.1371 | 0.0574 | 0.0667 | 0.0854 | 0.1247 | 0.0468 | 0.0564 | 0.0452 | 0.0692 | 0.0234 | 0.0292 | 0.0442 | 0.0680 | 0.0241 | 0.0299 | 0.1588 | 0.2356 | 0.0796 | 0.0985 |
| LLM2Rec | 0.1004 | 0.1403 | 0.0592 | 0.0691 | 0.0896 | 0.1299 | 0.0493 | 0.0594 | 0.0470 | 0.0718 | 0.0244 | 0.0305 | 0.0461 | 0.0691 | 0.0253 | 0.0312 | 0.1598 | 0.2367 | 0.0827 | 0.1018 |
| LLM_Vanilla | 0.0816 | 0.1142 | 0.0457 | 0.0540 | 0.0645 | 0.0985 | 0.0337 | 0.0425 | 0.0341 | 0.0563 | 0.0164 | 0.0220 | 0.0367 | 0.0575 | 0.0199 | 0.0250 | 0.1095 | 0.1654 | 0.0538 | 0.0678 |
| LLM_SFT | 0.1015 | 0.1407 | 0.0616 | 0.0715 | 0.0948 | 0.1365 | 0.0506 | 0.0612 | 0.0479 | 0.0724 | 0.0249 | 0.0310 | 0.0492 | 0.0716 | 0.0268 | 0.0325 | 0.1624 | 0.2374 | 0.0865 | 0.1052 |
| CLIP | 0.0735 | 0.1044 | 0.0408 | 0.0484 | 0.0663 | 0.0997 | 0.0358 | 0.0440 | 0.0344 | 0.0578 | 0.0173 | 0.0233 | 0.0357 | 0.0563 | 0.0191 | 0.0243 | 0.1033 | 0.1560 | 0.0506 | 0.0636 |
| VLM_Prompt | 0.0792 | 0.1123 | 0.0449 | 0.0530 | 0.0679 | 0.1008 | 0.0367 | 0.0449 | 0.0359 | 0.0594 | 0.0178 | 0.0235 | 0.0374 | 0.0588 | 0.0204 | 0.0256 | 0.1115 | 0.1668 | 0.0548 | 0.0690 |
| NoteLLM-2 | 0.1049 | 0.1438 | 0.0596 | 0.0690 | 0.0903 | 0.1309 | 0.0507 | 0.0608 | 0.0492 | 0.0744 | 0.0258 | 0.0320 | 0.0470 | 0.0730 | 0.0261 | 0.0324 | 0.1631 | 0.2396 | 0.0885 | 0.1072 |
| VLM_Vanilla | 0.0841 | 0.1178 | 0.0491 | 0.0577 | 0.0685 | 0.1031 | 0.0374 | 0.0463 | 0.0365 | 0.0596 | 0.0183 | 0.0240 | 0.0383 | 0.0601 | 0.0208 | 0.0262 | 0.1152 | 0.1695 | 0.0559 | 0.0695 |
| VLM_SFT | 0.1058 | 0.1461 | 0.0598 | 0.0699 | 0.0922 | 0.1349 | 0.0515 | 0.0622 | 0.0499 | 0.0755 | 0.0260 | 0.0325 | 0.0469 | 0.0738 | 0.0260 | 0.0326 | 0.1649 | 0.2365 | 0.0902 | 0.1082 |
| MLLM-MSR | 0.1038 | 0.1421 | 0.0611 | 0.0706 | 0.0940 | 0.1340 | 0.0506 | 0.0604 | 0.0482 | 0.0730 | 0.0253 | 0.0316 | 0.0494 | 0.0725 | 0.0272 | 0.0329 | 0.1627 | 0.2402 | 0.0864 | 0.1057 |
| I-LLMRec | 0.1064 | 0.1442 | 0.0624 | 0.0717 | 0.0967 | 0.1390 | 0.0539 | 0.0644 | 0.0483 | 0.0759 | 0.0258 | 0.0328 | 0.0503 | 0.0737 | 0.0278 | 0.0337 | 0.1644 | 0.2461 | 0.0881 | 0.1085 |
| **VLM2Rec_plug** | 0.1169* | 0.1588* | 0.0684* | 0.0786* | 0.1080* | 0.1511* | 0.0618* | 0.0728* | 0.0540* | 0.0823* | 0.0276* | 0.0345* | 0.0566* | 0.0849* | 0.0307* | 0.0378* | 0.1782* | 0.2718* | 0.0977* | 0.1216* |
| Imp.(%) | 9.87 | 8.69 | 9.62 | 9.62 | 11.69 | 8.71 | 14.66 | 13.04 | 8.22 | 8.43 | 6.15 | 5.18 | 12.52 | 15.04 | 10.43 | 12.17 | 8.07 | 10.44 | 8.31 | 12.07 |

## 7. Target numbers — secondary results

### 7.1 Dropout test, VLM2Rec setting, N@20 (Fig. 1 left; **read off the bar chart, ±0.001**)
| Data | Model | TEXT | VISION | FUSION | Gap |
|---|---|---|---|---|---|
| Beauty | VLM_Vanilla | 0.045 | 0.042 | 0.0442 | 6.9% |
| Beauty | VLM_SFT | 0.072 | 0.0645 | 0.0695 | 12.1% |
| Beauty | VLM2Rec | 0.078 | 0.077 | 0.0784 | 1.3% |
| ML-1M | VLM_Vanilla | 0.070 | 0.065 | 0.0711 | 7.2% |
| ML-1M | VLM_SFT | 0.114 | 0.0995 | 0.1087 | 14.6% |
| ML-1M | VLM2Rec | 0.1225 | 0.120 | 0.1252 | 2.0% |
Gradient-dynamics x-axis runs to ≈12k steps (Beauty) and ≈9k (ML) — not consistent with one-sample-per-user × 3 epochs at batch 32 or 8; unexplained (see DECISIONS D-02).

### 7.2 Geometry U (↓) / S (↑) (Tab. "ali_uni_clo" + App.)
| Data | Mod. | U Van | U SFT | U Ours | S Van | S SFT | S Ours |
|---|---|---|---|---|---|---|---|
| Beauty | Fused | -0.3984 | -1.7969 | -1.9921 | 1.0340 | 1.1910 | 1.1870 |
| Beauty | Vision | -0.0237 | -0.0398 | -0.2031 | 1.0071 | 1.0000 | 1.2134 |
| Beauty | Text | -0.3691 | -1.7422 | -1.6718 | 1.0381 | 1.1887 | 1.1639 |
| ML | Fused | -0.2275 | -0.8203 | -4.6562 | 1.0117 | 1.2471 | 1.2263 |
| ML | Vision | -0.0236 | -0.0078 | -1.7187 | 1.0035 | 0.9998 | 1.1134 |
| ML | Text | -0.2177 | -0.7891 | -1.1788 | 1.0118 | 1.0116 | 1.1135 |
| Toys | Fused | -0.4707 | -2.2500 | -1.7969 | 1.0609 | 1.2418 | 1.1609 |
| Toys | Vision | -0.0237 | -0.0039 | -0.5703 | 1.0039 | 0.9911 | 1.1551 |
| Toys | Text | -0.4453 | -2.2188 | -0.9335 | 1.0673 | 1.2480 | 1.1218 |
| Clothing | Fused | -0.4336 | -1.8359 | -1.9951 | 1.0444 | 1.0474 | 1.1274 |
| Clothing | Vision | -0.2368 | -0.0237 | -1.2136 | 1.0110 | 0.9997 | 1.0706 |
| Clothing | Text | -0.4102 | -1.7187 | -1.3759 | 1.0551 | 1.0562 | 1.0610 |

### 7.3 Ablation, N@20 (Fig. "ablation"; **read off chart, ±0.001**)
| Setting | Data | Ours | w/o WPCL | WPCL w/o StopGrad | w/o w_pen | w/o CRTR | VLM_SFT (line) |
|---|---|---|---|---|---|---|---|
| VLM2Rec | Beauty | 0.0785 | 0.0652 | 0.0729 | 0.0716 | 0.0743 | 0.0695 |
| VLM2Rec | ML-1M | 0.1253 | 0.1018 | 0.1149 | 0.1131 | 0.1167 | 0.1087 |
| plug (SASRec) | Beauty | 0.0728 | 0.0600 | 0.0665 | 0.0651 | 0.0673 | 0.0622 |
| plug (SASRec) | ML-1M | 0.1216 | 0.0990 | 0.1106 | 0.1090 | 0.1149 | 0.1082 |
Note: "w/o WPCL" < VLM_SFT but ≫ VLM_Vanilla, so it still injects CF somehow — exact definition unclear (D-21).

### 7.4 Generic gradient modulation, plug SASRec N@20 (Tab. "gradient_balancing")
| Data | Method | TEXT | VISION | FUSION | Gap% |
|---|---|---|---|---|---|
| Beauty | VLM_SFT | 0.0650 | 0.0587 | 0.0622 | 10.7 |
| Beauty | OGM-GE | 0.0677 | 0.0616 | 0.0632 | 9.9 |
| Beauty | AGM | 0.0664 | 0.0605 | 0.0648 | 9.8 |
| Beauty | WPCL only | 0.0667 | 0.0638 | 0.0673 | 4.5 |
| ML | VLM_SFT | 0.1100 | 0.0953 | 0.1082 | 15.4 |
| ML | OGM-GE | 0.1087 | 0.0965 | 0.1068 | 12.6 |
| ML | AGM | 0.1120 | 0.1008 | 0.1108 | 11.1 |
| ML | WPCL only | 0.1136 | 0.1108 | 0.1148 | 2.5 |
Head-SFT control (plug, N@20): Beauty Vanilla 0.0469/0.0438/0.0463 (7.08%), Head-SFT 0.0467/0.0385/0.0462 (21.30%); ML Vanilla 0.0665/0.0622/0.0695 (6.91%), Head-SFT 0.0950/0.0751/0.0916 (26.50%).

### 7.5 Additional backbones, plug N@20 (App. Tab. "add_backbones")
| Backbone | Data | LLM2Rec | LLM_SFT | VLM_Van | VLM_SFT | MLLM-MSR | I-LLMRec | **VLM2Rec_plug** |
|---|---|---|---|---|---|---|---|---|
| DiffuRec | Toys | 0.0639 | 0.0644 | 0.0586 | 0.0655 | 0.0650 | 0.0658 | 0.0760 |
| DiffuRec | Beauty | 0.0656 | 0.0670 | 0.0561 | 0.0682 | 0.0674 | 0.0689 | 0.0740 |
| DiffuRec | Clothing | 0.0316 | 0.0325 | 0.0291 | 0.0334 | 0.0323 | 0.0327 | 0.0384 |
| DiffuRec | Sports | 0.0319 | 0.0334 | 0.0288 | 0.0322 | 0.0340 | 0.0347 | 0.0398 |
| DiffuRec | ML-1M | 0.1036 | 0.1071 | 0.0767 | 0.1096 | 0.1077 | 0.1112 | 0.1188 |
| FMLP | Toys | 0.0639 | 0.0660 | 0.0534 | 0.0642 | 0.0664 | 0.0670 | 0.0732 |
| FMLP | Beauty | 0.0552 | 0.0564 | 0.0432 | 0.0585 | 0.0569 | 0.0598 | 0.0691 |
| FMLP | Clothing | 0.0265 | 0.0269 | 0.0206 | 0.0281 | 0.0264 | 0.0285 | 0.0341 |
| FMLP | Sports | 0.0272 | 0.0278 | 0.0225 | 0.0284 | 0.0280 | 0.0285 | 0.0353 |
| FMLP | ML-1M | 0.0844 | 0.0867 | 0.0582 | 0.0913 | 0.0876 | 0.0909 | 0.1057 |
| GRU4Rec | Toys | 0.0447 | 0.0454 | 0.0424 | 0.0465 | 0.0463 | 0.0475 | 0.0576 |
| GRU4Rec | Beauty | 0.0470 | 0.0476 | 0.0422 | 0.0485 | 0.0478 | 0.0491 | 0.0548 |
| GRU4Rec | Clothing | 0.0231 | 0.0233 | 0.0204 | 0.0242 | 0.0231 | 0.0239 | 0.0264 |
| GRU4Rec | Sports | 0.0267 | 0.0279 | 0.0216 | 0.0273 | 0.0283 | 0.0289 | 0.0305 |
| GRU4Rec | ML-1M | 0.0829 | 0.0867 | 0.0587 | 0.0902 | 0.0876 | 0.0896 | 0.0965 |

### 7.6 Hyperparameter sensitivity, plug SASRec N@20 (Fig. "hyperparameter"; read off chart)
- τ_WPCL: 0.05 best on Toys (0.0786), Beauty (0.0728), ML (0.1216), Sports, Clothing — i.e. **0.05**.
- τ_CRTR: Toys best 0.001; Beauty best 0.1; ML best 0.01; spread small (≤ ~0.003).
- λ: Toys/Beauty best 0.1; ML best 1.0; Sports 0.5; Clothing 1.0 (tiny differences).
- Official code defaults: τ_WPCL (`temp_cf`)=0.05, τ_CRTR (`temp_aux`)=0.1, λ (`lmd`)=0.1.

### 7.7 Hybrid SR models, N@20 (Tab. "modality_rec") — Tier 3
UniSRec (Clothing/ML): BERT .0308/.1086, LLM_SFT .0325/.1119, VLM_SFT(Text) .0328/.1124, I-LLMRec(Text) .0318/.1128, **VLM2Rec_plug(Text) .0336/.1137**.
TedRec: BERT .0332/.1167, LLM_SFT .0348/.1180, VLM_SFT(Text) .0355/.1189, I-LLMRec(Text) .0355/.1184, **.0362/.1208**.
M3SRec: CLIP .0322/.1134, VLM_SFT .0338/.1158, I-LLMRec .0347/.1165, **.0352/.1176**.
HM4SR: CLIP .0277/.1219, VLM_SFT .0328/.1261, I-LLMRec .0345/.1259, **.0396/.1315**.

### 7.8 Penalty variants, VLM2Rec setting N@20 (Tab. "penalty_variants")
| Penalty | Clothing | ML |
|---|---|---|
| None (w≡1, i.e. SFT+CRTR) | 0.0342 | 0.1131 |
| Static w=1+log2 | 0.0347 | 0.1145 |
| Batch-level | 0.0352 | 0.1212 |
| Modality-fixed (vision weak) | 0.0363 | 0.1236 |
| **User-adaptive** | **0.0370** | **0.1252** |

### 7.9 User-level (Tab. "user_stratification", "user_anchor_split")
Groups from Δgap under VLM_Vanilla: Low = bottom 20%, High = top 20%, Collapse = half of Low whose gap grows most under SFT.
- ML: Low 0.4048→0.0585 (red. 94.6%), Collapse 0.6322→0.0691 (100%), High 0.3642→0.0679 (92.5%). Pearson r (gap recovery vs rank recovery) weak .16/.25/.28, strong ≤.05, fused ≤.06.
- Clothing: Low .1711→.0473 (84.1%), Collapse .2800→.0561 (99.3%), High .1696→.0566 (80.4%); r weak .34/.43/.46.
- Anchor weak modality share: ML text 27.3% / vision 72.7%; Clothing text 46.2% / vision 53.8%.

### 7.10 Other
- VLM backbones dropout test (Beauty, plug, N@20), SFT gap → VLM2Rec gap: InternVL3-2B 10.04→2.48; Qwen2.5-VL-7B 8.16→1.77; LLaVA-1.5-7B 9.27→1.10; InternVL3-8B 8.04→2.34; Qwen2.5-VL-32B 6.70→2.28. FUSION (VLM2Rec): .0702/.0754/.0742/.0750/.0776.
- Qwen3-VL-Embedding (plug N@20, FUSION Base→+VLM2Rec): Beauty 2B .0719→.0746, 8B .0729→.0751; ML 2B .1147→.1162, 8B .1186→.1192.
- Negative sampling for Δgap (plug N@20; In-batch / Random / SASRec top-k / Text sim / Image sim): Clothing .0345/.0342/.0352/.0357/.0360; ML .1216/.1193/.1218/.1222/.1229.
- Item-embedding time (s, once per catalog): Beauty VLM_SFT 113, VLM2Rec_plug 113; Toys 79/79 (LLM methods ~38–58).
- Few-shot (plug SASRec N@20; read off chart): Toys K_u=256 .0689, 512 .0707, 1024 .0721, full .0786 (~75 min/epoch); Beauty 256 .0635, 512 .0651, 1024 .0662, full .0728 (~93 min/epoch). K_u=1024 ≈ 5–7 min/epoch.
- Cold-start (Beauty, plug N@20, groups 0 / 1–10 / 11–30 / 31+; read off chart): VLM2Rec_plug ≈ .022/.053/.077/.098; SASRec ≈ .015/.033/.065/.083.
- Cross-domain (plug N@20): Clothing→Beauty VLM2Rec_plug ≈ .059 (LLM_SFT .049); Sports→Clothing ≈ .029 (LLM_SFT .023).
- Case study (ML user 2289, target "Legends of the Fall"): rank Vanilla 22 → SFT 75 → VLM2Rec 3.
