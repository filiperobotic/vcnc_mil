# Step 5 — Validation Plan (VCNC gate + OA-MIL integration)

Combinação do `VCNCKMeansConfusionAwareHook` (configuração FIXED,
`action='filter'`) com o pipeline OA-MIL completo. Sem modificação de
código: somente configs novos.

## 1. Arquivos criados

| Arquivo | Propósito |
|---|---|
| `custom_configs/exp_configs/exp_vcnc_gate_oamil_smoke.py` | Smoke test ~3 epochs. Herda de `exp_oamil_voc_sim40_oaie_early.py` (oaie_epoch=2). VCNC com `progressive_epochs=2` → gate dispara primeiro na epoch 3. |
| `custom_configs/exp_configs/exp_vcnc_gate_oamil_full.py` | Produção, 12 epochs. Herda de `exp_oamil_voc_sim40.py` (oaie_epoch=9). VCNC com `progressive_epochs=4` → gate dispara primeiro na epoch 5. |

Nenhum dos arquivos abaixo foi tocado: `standard_roi_head_oamil.py`,
`shared2fc_bbox_head_oamil.py`, `vcnc_kmeans_confusion_aware_spatial_hook.py`,
`oamil_epoch_injector_hook.py`.

## 2. O que cada config faz

### Smoke (`exp_vcnc_gate_oamil_smoke.py`)

Linha do tempo (epoch 1-indexed):

| Epoch | OA-MIL | VCNC | Notas |
|---|---|---|---|
| 1 | gates fechados (oais/oaie_epoch=2 → `_epoch+1=1<2`) | `Warmup, pulando` (warmup_epochs=1) | comportamento ≈ baseline; sanity log do OA-MIL **não** dispara ainda |
| 2 | OA-IS + loss_oais + OA-IE ativos (`_epoch+1=2>=2`) | E2 + E3 rodam; gate **não** ativo (progressive_epochs=2, `aggressive_only=True` exige `epoch > 2`) | sanity log dispara **uma vez** no primeiro batch; espera-se `n_oaie_iters=5` |
| 3 | OA-MIL contínuo | E2 + E3 **+ gate (primeira vez)** | banner `[VCNC-Spatial] === GATE DE CONFUSÃO ===` no log |

### Full (`exp_vcnc_gate_oamil_full.py`)

12 epochs, paper-equivalent:

| Epoch | OA-MIL | VCNC | Notas |
|---|---|---|---|
| 1 | gates fechados | warmup | baseline |
| 2 | OA-IS + loss_oais (sem OA-IE) | E2 + E3 (sem gate) | sanity log com `n_oaie_iters=1` |
| 3-4 | OA-IS + loss_oais (sem OA-IE) | E2 + E3 (sem gate) | gate ainda em fase progressiva |
| 5 | OA-IS + loss_oais (sem OA-IE) | E2 + E3 **+ gate (primeira vez)** | banner GATE |
| 6-8 | continua | continua | |
| 9 | OA-IS + loss_oais **+ OA-IE liga** | continua | sanity log **não** dispara de novo (já foi feito na epoch 2); `loss_oais` salta de fórmula simples pra `refine` (provável aumento na primeira iter da epoch 9) |
| 10-12 | full pipeline | continua | |

⚠️ Nota: a sanity log do OA-MIL dispara na primeira ativação de
`oamil_active` (epoch 2). Quando OA-IE liga na epoch 9, **não há novo
sanity log**, porque a flag `self._sanity_printed` foi setada antes.
Para confirmar OA-IE liga, grepar `n_oaie_iters` no `[OA-MIL sanity]`
da epoch 2 **não** reflete OA-IE (ele é 1 nessa epoch porque
`oaie_epoch=9`). Em vez disso, observar o salto no valor de
`loss_oais` na epoch 9.

(O config `_full` é diagnóstico; o `_smoke` é onde o sanity log do
OA-MIL captura `n_oaie_iters=5` por ter `oaie_epoch=2`.)

## 3. Ordem e dependência entre os hooks

Investigado:

- `OAMILEpochInjectorHook.before_train_epoch`: apenas
  `runner.model.roi_head.bbox_head._epoch = runner.epoch`. Pure Python
  attribute write. Custo: O(1).
- `VCNCKMeansConfusionAwareHook.before_train_epoch`: coleta predições,
  K-means c30, computa matriz de confusão, decide pares "blacklisted",
  E2/E3 (relabel por clustering, spatial refinement), opcionalmente
  E4. Toca **anotações do dataset**, não toca `bbox_head` nem
  `roi_head`. Custo: segundos.

**Dependência funcional**: nenhuma. Verificado:

- O VCNC hook lê/escreve dataset annotations + roda `model.eval()` /
  `model.train()` internamente. Não toca `bbox_head._epoch`.
- O OA-MIL injector só seta `bbox_head._epoch`. Não toca dataset.

Ambos com `priority='NORMAL'` (o VCNC herda do `Hook` base; o OA-MIL
injector seta `'NORMAL'` explicitamente). Quando dois hooks com a
mesma prioridade implementam o mesmo entry point, mmengine executa na
ordem em que aparecem em `custom_hooks`.

**Ordem escolhida** (intencional, mas não obrigatória):

```python
custom_hooks = [
    dict(type='OAMILEpochInjectorHook'),     # O(1), roda primeiro
    dict(type='VCNCKMeansConfusionAwareHook', ...),  # caro, roda depois
]
```

Pode inverter sem mudança funcional. A ordem só importa se um dia um
dos hooks ler estado escrito pelo outro — hoje não é o caso.

## 4. Confirmação do sanity log do OA-MIL

O sanity log do OA-MIL (em `_oamil` no `standard_roi_head_oamil.py`)
dispara quando:

1. `self._sanity_printed` é `False` (lazy init via `getattr`).
2. `oamil_active=True` (gate de epoch aberto + `oamil_lambda > 0`).
3. `oais_info is not None` (há positivos no batch).

Nada disso depende do VCNC hook. O VCNC modifica annotations
**antes** do epoch começar, mas isso só muda o conteúdo dos batches —
não as flags do bbox_head nem o ciclo do `_oamil`. Portanto, o sanity
log continua disparando **uma vez** no primeiro batch da epoch 2:

```
[OA-MIL sanity] confidence_scores.requires_grad=True,
                best_score.requires_grad=False,
                pseudo_bbox_targets.requires_grad=False,
                loss_oais.requires_grad=True,
                n_oaie_iters=5    ← smoke; 1 no full antes da epoch 9
```

## 5. Diagnóstico de "first fire" do gate VCNC

A escolha foi **não criar novo hook**, em respeito ao constraint
explícito de não modificar código existente. O hook VCNC **já imprime**
um banner detalhado na primeira (e em todas as) ativação do gate,
diretamente da linha 694 do
`vcnc_kmeans_confusion_aware_spatial_hook.py`:

```
[VCNC-Spatial] === GATE DE CONFUSÃO ===
[VCNC-Spatial] Pares válidos analisados: <N_valid>
[VCNC-Spatial] Pares marcados como ruidosos: <N_confused>
[VCNC-Spatial] Severidade — mediana: <m>, MAD: <mad>
[VCNC-Spatial] Threshold = max(med+3.0*MAD, 10.0*med) = <thr>
[VCNC-Spatial] Top 10 pares por severidade:
[VCNC-Spatial]   classes (i, j): sev=..., severity_zscore=...
...
```

Para a sua diagnóstica de "first fire", grepar a primeira ocorrência
do banner:

```bash
grep -n "=== GATE DE CONFUSÃO ===" work_dirs/step5_smoke/*/[0-9]*.log | head -1
```

E pra extrair epoch + pares juntos no estilo que você queria:

```bash
awk '/========== Época / {ep=$0} /=== GATE DE CONFUSÃO ===/ {print ep; getline; print; exit}' \
    work_dirs/step5_smoke/*/[0-9]*.log
```

Saída esperada (smoke):
```
[VCNC-Spatial] ========== Época 3 ==========
[VCNC-Spatial] === GATE DE CONFUSÃO ===
[VCNC-Spatial] Pares válidos analisados: <N>
```

Se quiser **mesmo** um log dedicado no formato pedido
(`[VCNC gate] First fire at epoch X, pairs detected: N`), isso exigiria
um novo arquivo de hook (observer) — fora do escopo "apenas crie os
configs novos". Posso adicionar num passo separado se você priorizar.

## 6. Comandos sugeridos

### Smoke (3 epochs)

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil

python tools/train.py \
    custom_configs/exp_configs/exp_vcnc_gate_oamil_smoke.py \
    --work-dir work_dirs/step5_smoke \
    --cfg-options \
        randomness.deterministic=True \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=999
```

Pra greppar os 3 marcadores principais (warmup, sanity OA-MIL, first-fire gate):

```bash
WD=work_dirs/step5_smoke/*/[0-9]*.log
grep -E "Warmup, pulando|========== Época|\[OA-MIL sanity\]|=== GATE DE CONFUSÃO ===" $WD
```

### Full (12 epochs)

```bash
python tools/train.py \
    custom_configs/exp_configs/exp_vcnc_gate_oamil_full.py \
    --work-dir work_dirs/step5_full \
    --cfg-options \
        randomness.deterministic=True
```

(Sem cap em `max_epochs`; vai rodar os 12 do schedule.)

## 7. O que esperar nos logs (smoke)

| Marcador | Quando | Conteúdo |
|---|---|---|
| `[VCNC-Spatial] Época 1: Warmup, pulando.` | epoch 1 | VCNC pula |
| `[VCNC-Spatial] ========== Época 2 ==========` | epoch 2 | VCNC roda E2/E3 (sem gate) |
| `[OA-MIL sanity] ...` | epoch 2, primeiro batch | requires_grad esperado + `n_oaie_iters=5` |
| `loss_oais` na key do log | epoch 2+ | valor ≈ 0.09–0.16 conforme Step 4 |
| `[VCNC-Spatial] ========== Época 3 ==========` | epoch 3 | VCNC roda tudo |
| `[VCNC-Spatial] === GATE DE CONFUSÃO ===` | epoch 3 | **first fire do gate** |

## 8. Pontos de conflito identificados

| Risco | Avaliação | Mitigação |
|---|---|---|
| Ordem dos hooks afeta resultado | Não — atributos lidos/escritos são disjuntos | N/A |
| `reload_dataset=True` do VCNC interfere com sampler do dataloader | Possível pequeno descompasso de ordem de batches após reload, mas seed do runner é estável → comportamento determinístico dentro do mesmo config | N/A |
| `model.eval()` interno do VCNC interfere com BN durante seu pipeline | VCNC restaura `model.train()` antes de retornar (verificado nos prints da linha 991 — "Resumo Época" depois do ciclo) | Confirmar manualmente se vir bug |
| OA-MIL roda no batch ENQUANTO VCNC processa epoch | Não — VCNC roda **antes** do epoch começar (`before_train_epoch`), termina, e só depois o batch loop começa. OA-MIL roda **dentro** do batch loop | N/A |
| Confusion gate "filter" marca pares com `ignore_flag=1`, modificando o conjunto de positivos do sampler — OA-MIL pode ficar com bags diferentes | É a interação **desejada**: o gate remove pares de classes confusas, e OA-MIL refina caixas das restantes. Validar na prática | Esperar `loss_oais` levemente diferente do sim40-só-OA-MIL |
| Sanity log do OA-MIL roda só uma vez — não captura mudança de OA-IE na epoch 9 (config full) | É by-design (`_sanity_printed` lazy flag) | Para inspecionar OA-IE no full, ler valor de `loss_oais` antes/depois da epoch 9 |
| `custom_imports` no novo config substitui o do base | Sim — mmengine REPLACE em vez de merge para `custom_imports`. Por isso listei TODOS os imports (4 caminhos) explicitamente | N/A se a lista estiver completa |

## 9. Smoke checklist (você fazendo)

- [ ] Epoch 1: log mostra `Warmup, pulando` (VCNC) e nenhum `[OA-MIL sanity]`.
- [ ] Epoch 2: primeiro batch tem `[OA-MIL sanity]` com `n_oaie_iters=5`,
      `loss_oais` aparece em iters subsequentes.
- [ ] Epoch 3: banner `[VCNC-Spatial] === GATE DE CONFUSÃO ===` aparece
      antes do epoch loop começar.
- [ ] Sem NaN, sem crash. `loss_bbox` divergente do baseline (esperado
      pela combinação de pseudo-GT + annotations filtradas pelo gate).
- [ ] Tempo por epoch ≈ Step 4 + overhead VCNC (~10-30s pra clustering +
      relabel + reload de annotations).

## 10. Próximo passo

Quando o smoke passar:
- Rodar o `_full` (12 epochs) → comparação contra:
  - Baseline puro (`_diag_clean_baseline.py` ou
    `exp5_12x1[9]_baseline_danieldb_sim40.py`).
  - Só OA-MIL (`exp_oamil_voc_sim40.py`).
  - Só gate (`exp5_12x1[9[_vcnc_keans30_confustion_aware_gatefilter_spatial_sim40_fixed.py`).
- Análise: ganho da combinação > ganho de cada um isoladamente?

**Não rodei treino.** Configs prontos no disco.
