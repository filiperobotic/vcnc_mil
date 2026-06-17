# Step 4 — Validation Plan

OA-IE implementado. Pipeline OA-MIL completo (OA-IS pseudo, `loss_oais`
agregação MIL, OA-IE iteração múltipla) — todos os 3 gates independentes.

## 1. Arquivos modificados / criados

| Arquivo | Mudança |
|---|---|
| `custom_configs/models/standard_roi_head_oamil.py` | (1) Removida a `NotImplementedError` do OA-IE. (2) `_oamil_instance_selection` agora roda loop de `(oaie_num + 1)` iterações quando `compute_oaie=True`; com `compute_oaie=False`, 1 iteração — comportamento Step 3. (3) `oais_info` carrega agora `scores_list` (lista de tensores por iteração), `bags`, `pos_labels`, `_sample_best_score`. (4) Novo `_combine_inst_scores` (estático) aplica a fórmula `refine` / `random`. (5) `_aggregate_instance_scores` mudou de assinatura: agora recebe `(scores, bags, pos_labels)` separadamente — chamado uma vez por iteração. (6) `_oamil_log_sanity` aceita `n_iters` e o imprime. (7) `_oamil` valida `oaie_type` e `oaie_num >= 1` quando OA-IE ativa. |
| `custom_configs/exp_configs/exp_oamil_voc_sim40.py` | `oaie_flag` de `False` → `True`. `oaie_epoch=9` mantido (paper). |
| `custom_configs/exp_configs/exp_oamil_voc_sim40_oaie_early.py` | **Novo**. Herda do production via `_base_` e sobrescreve **só** `oaie_epoch=2`. Pra validação rápida em 3 epochs. |

`shared2fc_bbox_head_oamil.py` **não muda** — contrato do `pseudo_bbox_targets` já é o mesmo.

## 2. Estrutura do loop OA-IE

Mapa linha-a-linha vs ref OA-MIL 2.x (linhas 246–268):

| Etapa | Ref OA-MIL 2.x | Step 4 (nosso) |
|---|---|---|
| `iter_num` | linha 247 | `(oaie_num + 1) if compute_oaie else 1` |
| Loop | `for i in range(iter_num)` | mesmo |
| Pega `bbox_pred` corrente | linha 250 | `current_bbox_pred` (mantida fora do loop) |
| Filtra por classe | linha 252 | `current_bbox_pred = bbox_pred.view(...)[arange, pos_labels]` no fim do iter |
| Decode source: iter 0 ou random | linhas 255–258 | `if i == 0 or oaie_type == 'random': decode_source = pos_rois_xyxy else: decode_source = current_roi[:, 1:]` |
| Decode | linha 256 ou 258 | `bbox_coder.decode(decode_source, current_bbox_pred).clamp(min=0.0)` |
| Build new_roi | linhas 260–261 | mesmo |
| Append boxes | linha 262 | `oaie_bboxes_list.append(...)` |
| Segundo forward | linha 266 | `self._bbox_forward(x, current_roi)` |
| Scores | linha 267 | `softmax(cls_score)[arange, pos_labels]` |
| Append scores | linha 268 | `oaie_scores_list.append(scores)` |
| Update `current_bbox_pred` | implícito no `bbox_pred = new_bbox_results['bbox_pred']` no topo do próximo iter | explícito no fim do iter via `current_bbox_pred = ...` |

Diferença estilística: o ref reusa `new_bbox_results` como variável que
"flutua" iter-a-iter; eu uso `current_bbox_pred` como variável de loop
explícita (mesma semântica, mais legível).

**Pseudo-GT continua usando iter 0** (`oaie_bboxes_list[0]`,
`oaie_scores_list[0]`) — corresponde à linha 280 do ref.

## 3. Fórmulas de agregação

`_combine_inst_scores(inst_scores_list, oaie_type, oaie_num, oaie_coef)`:

```
len == 1           : loss = 1 − s₀                                (Step 3 caminho)
len > 1, 'refine'  : loss = (1 − s₀) + (1 − mean(s₁..s_N)) · oaie_coef
len > 1, 'random'  : loss = 1 − mean(s₀..s_N)
```

Então `loss_oais = loss · oamil_lambda`.

Para o config `_oaie_early.py` (`oaie_num=4`, `oaie_type='refine'`,
`oaie_coef=1.0`):

```
loss_oais = (1 − s₀ + 1 − (s₁+s₂+s₃+s₄)/4) · 0.1
          = ((1 − s₀) + (1 − mean(s₁..s₄))) · 0.1
```

`s₀` é o mesmo scalar que rolou na Step 3 (mesmo segundo forward).

## 4. Fluxo de gradiente entre iterações

Não é tão limpo quanto Step 3. Cada iteração:

```
i=0:  new_pred_boxes_0 = decode(pos_rois_xyxy, pos_bbox_pred)
                          ↑ pos_rois sem grad     ↑ vem do primeiro forward (com grad)
      new_roi_0[:, 1:] = new_pred_boxes_0   → new_roi_0 herda grad
      new_bbox_results_0 = _bbox_forward(x, new_roi_0)  ← RoIAlign é diferenciável wrt new_roi_0
      scores_0 = softmax(new_bbox_results_0)[..., labels]

i=1+ ('refine'):
      decode_source = new_roi_(i-1)[:, 1:]    ← carries grad from prev iter
      new_pred_boxes_i = decode(decode_source, current_bbox_pred)
                          ↑ has grad from prev iter   ↑ from prev iter's forward
      ... (forward, scores) ...
```

Cada iteração soma um forward através do `bbox_head`. No backward,
o gradiente flui por **todas** as iterações encadeadas:

```
loss_oais → cls_score (iter N) → bbox_head fc → bbox_feats (iter N)
                                                → x (feature maps) ← compartilhado
                              → new_roi (iter N) → new_pred_boxes (iter N)
                                                 → decode(new_roi (N-1), bbox_pred (N-1))
                                                 ↓ recursão
                                                 → ... iter N-1 → ... → iter 0
                                                                       → bbox_pred (primeiro forward)
                                                                       → bbox_feats (primeiro forward)
                                                                       → x ← compartilhado
```

Para 'random': `decode_source` é sempre `pos_rois_xyxy` (sem grad), o que
desacopla o decode mas o `current_bbox_pred` continua trazendo grad da
iter anterior. Encadeamento mais raso, ainda existe.

**Custo de memória**: ~5x o forward do `bbox_head` (2 FCs + 2 cabeças)
ativos no autograd. Aceitável. O backbone é compartilhado.

**Custo de tempo por iter de treino**: ~50–100% a mais do que Step 3
(esperado, 4 forwards extras + 4 backwards extras no `bbox_head`).

## 5. Asserts e checagens defensivas

Tudo da Step 3 mantido. Novos:

```python
assert bbox_head.oaie_type in ('refine', 'random')
```

Em `_oamil`, antes de chamar a pipeline com OA-IE ativo. Falha cedo se
alguém digitar `oaie_type='unknown'`.

```python
assert bbox_head.oaie_num >= 1
```

OA-IE com `oaie_num=0` quer dizer `iter_num=1` — equivalente a OA-IE
desligada. Falha cedo pra evitar config silenciosamente sem efeito.

```python
assert len(oaie_bboxes_list) == iter_num
assert len(oaie_scores_list) == iter_num
```

Sentinelas no fim do loop.

```python
assert len(inst_scores_list) == oaie_num + 1
```

Dentro de `_combine_inst_scores`, no caminho `len > 1`. Pega
inconsistência entre `iter_num` real e `oaie_num` do config.

## 6. Sanity log expandido

Mesmo formato da Step 3, mas com um campo extra:

```
[OA-MIL sanity] confidence_scores.requires_grad=True,
                best_score.requires_grad=False,
                pseudo_bbox_targets.requires_grad=False,
                loss_oais.requires_grad=True,
                n_oaie_iters=5
```

Para o config `_oaie_early.py`, `n_oaie_iters` deve ser **5** (`oaie_num + 1`).
Para um config sem OA-IE (Step 3 prod equivalente), seria **1**.

`confidence_scores.requires_grad` agora é lido de `scores_list[0]`
(primeira iteração). Continua `True` — mesmo princípio da Step 3.

## 7. Comandos sugeridos pra você validar

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil

# Run de referência
python tools/train.py \
    custom_configs/exp_configs/_diag_clean_baseline.py \
    --work-dir work_dirs/step4_ref \
    --cfg-options \
        randomness.deterministic=True \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=999

# Run com OA-MIL completo (OA-IE early)
python tools/train.py \
    custom_configs/exp_configs/exp_oamil_voc_sim40_oaie_early.py \
    --work-dir work_dirs/step4_oaie_early \
    --cfg-options \
        randomness.deterministic=True \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=999
```

Para grep do sanity:

```bash
grep "OA-MIL sanity" work_dirs/step4_oaie_early/*/[0-9]*.log
```

## 8. O que esperar nas comparações

### Epoch 1 (`_epoch=0`)

Gates de OA-IS, `loss_oais` e OA-IE **fechados** (`epoch_open=False`).
Comportamento idêntico ao Step 1. `loss_oais` ausente, `loss_bbox`
casa com baseline (ruído CUDA). Sanity log **não dispara** ainda.

### Epoch 2 abre (`_epoch=1`)

Com `oaie_epoch=2` no early config, **todos** os gates abrem juntos.

- Primeiro iter da epoch 2: sanity log dispara, `n_oaie_iters=5`.
- `loss_oais` aparece. Magnitude esperada:

  ```
  loss_oais = ((1 − s₀) + (1 − mean(s₁..s₄))) · 0.1
  ```

  Da Step 3 sabemos `s₀ ≈ 0.2` (inst_score depois de max-pooling sobre
  caixas decodadas pelo modelo recém-saído da epoch 1). Os scores
  `s₁..s₄` dependem da qualidade do refinamento:

  | Cenário | s₀ | mean(s₁..s₄) | loss_oais (sem λ) | * λ=0.1 |
  |---|---|---|---|---|
  | Refinamento bom (scores sobem rápido) | 0.2 | 0.5 | 0.8 + 0.5 = 1.3 | 0.13 |
  | Refinamento moderado | 0.2 | 0.3 | 0.8 + 0.7 = 1.5 | 0.15 |
  | Refinamento ruim (≈ não muda) | 0.2 | 0.2 | 0.8 + 0.8 = 1.6 | 0.16 |
  | Refinamento ótimo (quase 1) | 0.2 | 0.9 | 0.8 + 0.1 = 0.9 | 0.09 |

  **Expectativa realista**: `loss_oais ≈ 0.13–0.16` na abertura.
  Decresce conforme o modelo aprende.

  ⚠️ Nota: a expectativa do usuário ("Step 4 deve ser MENOR que Step 3
  ≈0.08") só vale no cenário ótimo de refinamento (`mean(s₁..s₄) ≈ 0.9`)
  com `oaie_coef < 1`. Com `coef=1.0` do paper e refinamento moderado,
  `loss_oais` Step 4 tende a ser **MAIOR** que Step 3 (porque há um
  termo aditivo). Se sair MENOR que Step 3 (~0.08), excelente — sinal
  de refinamento muito eficaz. Se sair em ~0.13–0.16, é o regime
  esperado pela matemática.

- `loss_bbox` continua divergente do baseline (mesma lógica da Step 2/3).
- `loss_cls`, `loss_rpn_*` próximos do baseline.
- Tempo por iter ~50–100% mais lento que Step 3 (4 forwards extras).

### Epoch 3

Continuação da epoch 2. `loss_oais` decresce gradualmente. Esperado
~10–25% de queda sobre a epoch 2.

## 9. Sinais de problema

| Sintoma | Provável causa |
|---|---|
| `AssertionError: unknown oaie_type` | Typo no config |
| `AssertionError: oaie_num must be >= 1` | Config com `oaie_num=0` mas `oaie_flag=True` |
| `AssertionError: inst_scores_list length ... does not match oaie_num+1` | Bug interno — desincronia entre `iter_num` e `oaie_num` |
| `sanity log com n_oaie_iters=1` na epoch 2 | OA-IE não disparou; verificar `oaie_flag=True` e `oaie_epoch≤2` no config carregado |
| `loss_oais` na epoch 2 = `(1 − s₀)·0.1` (mesmo valor da Step 3) | Mesmo problema — OA-IE não rodou |
| `loss_oais` na epoch 2 com valor > 0.5 | `oaie_coef` errado, ou scores muito ruins; investigar |
| OOM CUDA | Esperado: 5 iterações de bbox_head + RoIAlign no autograd; reduzir `oaie_num` se rolar |
| Treino ~3x mais lento | Aceitável — overhead do loop OA-IE |
| `loss_oais.requires_grad=False` no sanity | Bug — `scores_list[0]` perdeu grad em algum lugar; investigar |

## 10. Smoke checklist (você fazendo)

- [ ] Epoch 1: `loss_oais` ausente do log.
- [ ] Epoch 2 primeiro batch: sanity log impresso UMA vez com `n_oaie_iters=5`.
- [ ] Epoch 2+: `loss_oais` ≈ 0.09–0.16 inicialmente, decrescente.
- [ ] `loss_bbox` continua divergente do baseline; `loss_cls` ≈ baseline.
- [ ] Sem NaN, sem crash.
- [ ] Tempo por iter ~1.5–2x da Step 3 (verificar `time` no log).

## 11. Próximo passo

Quando passar:

- Production config `exp_oamil_voc_sim40.py` (com `oaie_epoch=9`) está
  pronta pra um treino completo de 12 epochs. Mesmas hiperparams do
  paper VOC.
- O pipeline OA-MIL está completo. Próximas linhas de trabalho
  prováveis: combo com `VCNCKMeansConfusionAwareHook` (gate de classes
  ruidosas + OA-MIL), benchmarks contra baselines, ablações.

Por enquanto: **aguardando seus runs**. Não rodei treino.
