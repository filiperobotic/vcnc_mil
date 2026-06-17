# Step 3 — Validation Plan

`loss_oais` (agregação MIL) implementado. OA-IE continua stubada.

## 1. Arquivos modificados

| Arquivo | O que mudou |
|---|---|
| `custom_configs/models/standard_roi_head_oamil.py` | (1) Removida a `NotImplementedError` para `oamil_lambda > 0`. (2) `_oamil_instance_selection` agora retorna **tupla** `(pseudo_bbox_targets, oais_info)` e aceita `compute_pseudo`. `oais_info` carrega `confidence_scores` (com grad), `bags`, `pos_labels`, e `_sample_best_score` (para o sanity log). (3) Novo método `_aggregate_instance_scores` (agregação MIL em 3 níveis). (4) `_oamil` calcula `loss_oais = (1 - inst_score) * oamil_lambda` quando o gate abre. (5) Novo `_oamil_log_sanity` (estático) imprime `requires_grad` dos 4 tensores-chave **uma única vez** via `MMLogger`, gated por `self._sanity_printed`. |
| `custom_configs/exp_configs/exp_oamil_voc_sim40.py` | `oamil_lambda` de `0.0` → `0.1` (valor VOC do paper). Demais campos OA-MIL inalterados. |

`shared2fc_bbox_head_oamil.py` **não muda** — o contrato dele
(`pseudo_bbox_targets` opcional) já cobre Step 3.

## 2. Agregação MIL — mapeamento linha-a-linha vs o ref

Método novo: `_aggregate_instance_scores(self, oais_info) -> Tensor`.
Equivalência com `_get_instance_cls_scores` do ref (linhas 306–325):

| Etapa | Ref OA-MIL 2.x | Step 3 (nosso) |
|---|---|---|
| Loop por bag | linhas 311–314 | `inst_scores = stack([confidence_scores[ib].max() for ib in bags])` |
| Label do bag | linha 314 (`pos_labels[inst_inds[0]]`) | `inst_labels = stack([pos_labels[ib[0]] for ib in bags])` |
| Cat instance lists | linhas 316–317 | `stack` (equivalente para escalares) |
| Loop por classe | linhas 321–323 | `for c in unique_cls: cls_scores_sum += inst_scores[inst_labels == c].mean()` |
| Divisão pelo nº de classes | linha 324 | `cls_scores_sum / unique_cls.numel()` |
| Retorno | linha 325 | mesmo (escalar 0-dim) |

Matematicamente, retorna:

```
inst_score = mean_c ( mean_{i in class c} ( max_{r in bag_i}( score_r ) ) )
```

E em `_oamil`:

```python
loss_oais = (1.0 - inst_score) * oamil_lambda
```

Branch único pela construção da Step 3: `inst_scores_list = [inst_score]`,
e como `len == 1`, cai direto no `loss_oais = (1 - inst_scores_list[0]) *
lambda`. A lista existe apenas pra deixar Step 4 (OA-IE) trivialmente
encaixável — vai poder fazer `.append(...)` em cada iteração e cair no
branch `len > 1` com `refine` / `random`.

## 3. Fluxo de gradiente

Mudança crítica vs Step 2:

- **Step 2**: as duas instâncias (`best_score`, `best_pred`) usadas para
  `pseudo_gt` foram `.detach()`-adas — `pseudo_bbox_targets` não tem
  grad e funciona apenas como **target** do `loss_bbox`.
- **Step 3**: `confidence_scores` (saída do segundo forward) **não é
  mais detached**. Ele alimenta `_aggregate_instance_scores`, onde
  `max → mean → mean` são todas operações diferenciáveis. O gradiente
  de `loss_oais` retropropaga por:

  ```
  loss_oais
    → cls_score (segundo forward)
       → bbox_head.fc_cls.weight/bias
       → bbox_head.shared_fcs[*]   (shared FC)
       → bbox_feats (segundo RoIAlign)
       → x (feature maps)          ← se quiser, vai até o backbone
  ```

  O OA-IS continua usando `.detach()` localmente em `best_score` e
  `best_pred_box` (linhas 159–161 da nova versão) — o `confidence_scores`
  original mantém o grad-state intacto para o aggregator.

Validação dessa separação no código: depois do segundo forward, eu
construo `oais_info` capturando `confidence_scores` **antes** de
qualquer detach. A branch OA-IS só faz detach das fatias específicas
(`best_score = bag_scores[best_pos].detach()`), o que cria um nó
detached **novo** sem afetar a referência original em `oais_info`.

## 4. Gates de ativação

```
oaie_active   = oaie_flag       AND _epoch+1 >= oaie_epoch  → NotImplementedError (Step 4)
epoch_open    = _epoch+1 >= oais_epoch
oamil_active  = oamil_lambda > 0 AND epoch_open
oais_active   = oais_flag       AND epoch_open

se nem oamil_active nem oais_active:
    → ({}, None)                              # comportamento Step 1

se um ou ambos ativos:
    → roda _oamil_instance_selection(compute_pseudo=oais_active)
    se oamil_active:
        → adiciona 'loss_oais' ao dict
```

Para nosso config (`oais_flag=True, oamil_lambda=0.1, oais_epoch=2,
oaie_flag=False`):

| epoch (1-indexed) | `_epoch` | `epoch_open` | `oamil_active` | `oais_active` | Resultado |
|---|---|---|---|---|---|
| 1 | 0 | 1 ≥ 2 → False | False | False | Step 1 behavior — sem `loss_oais`, sem pseudo |
| 2 | 1 | 2 ≥ 2 → True | True | True | OA-IS pseudo + `loss_oais` no dict |
| 3 | 2 | True | True | True | Mesmo da epoch 2 |

## 5. Asserts e checagens defensivas

Tudo que veio da Step 2 (`shape mismatch`, `reg_class_agnostic`,
`reg_decoded_bbox`, `isfinite(pseudo_bbox_targets)`) continua intacto.
**Novos** em Step 3:

```python
assert len(inst_scores_list) == 1, (
    'Step 3 must produce exactly one inst_score; OA-IE is still stubbed.')
```

Sentinela contra alguém ligar OA-IE acidentalmente. Como
`oaie_active=True` já levanta `NotImplementedError` antes, esse assert
é redundante na prática — fica como sanity check defensivo dentro do
mesmo método.

```python
if not torch.isfinite(loss_oais):
    raise RuntimeError(
        f'loss_oais became non-finite ({loss_oais.item()}). '
        'Check confidence_scores and oamil_lambda.')
```

Sync trivial por iter (1 escalar). Pega NaN/Inf imediatamente com
mensagem clara em vez de poluir o log com NaN propagado.

Edge case **sem positivos no batch**: `_oamil_instance_selection`
retorna `(None, None)`. Em `_oamil`, a chave `loss_oais` simplesmente
não é adicionada (a guarda `if oamil_active and oais_info is not None`
filtra). Mesmo comportamento que Step 2 — sem crash, sem chave
artificial.

### Sanity log de gradiente (one-shot)

Logado **uma única vez**, na primeira iteração em que `oamil_active=True`
(epoch 2, primeiro batch). Gate via `self._sanity_printed` (lazy-init
por `getattr`). Formato:

```
[OA-MIL sanity] confidence_scores.requires_grad=True,
                best_score.requires_grad=False,
                pseudo_bbox_targets.requires_grad=False,
                loss_oais.requires_grad=True
```

Valores esperados — qualquer outro indica bug na wiring:

| Tensor | Esperado | Por quê |
|---|---|---|
| `confidence_scores.requires_grad` | **True** | Sai do segundo forward sem detach; precisa fluir grad para `loss_oais` retropropagar até as FCs |
| `best_score.requires_grad` | **False** | `.detach()` explícito; só usado para `phi` que parametriza a fração de mistura (não deve criar gradiente) |
| `pseudo_bbox_targets.requires_grad` | **False** | Computado a partir de tensores detached + `bbox_coder.encode`; é tratado como constante por `loss_bbox` |
| `loss_oais.requires_grad` | **True** | Composição diferenciável a partir de `confidence_scores`; só faz sentido se for otimizável |

`MMLogger.get_current_instance()` escreve no log do mmengine, então
aparece no `*.log` do `work_dir` automaticamente, em nível INFO. Para
filtrar:

```bash
grep "OA-MIL sanity" work_dirs/step3_oais_loss/*/[0-9]*.log
```

## 6. Comandos sugeridos pra você validar

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil

# Run de referência (sem OA-MIL) — pra contraste em loss_cls / loss_rpn_*
python tools/train.py \
    custom_configs/exp_configs/_diag_clean_baseline.py \
    --work-dir work_dirs/step3_ref \
    --cfg-options \
        randomness.deterministic=True \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=999

# Run com OA-IS + loss_oais ativos
python tools/train.py \
    custom_configs/exp_configs/exp_oamil_voc_sim40.py \
    --work-dir work_dirs/step3_oais_loss \
    --cfg-options \
        randomness.deterministic=True \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=999
```

(Remova `val_interval=999` se quiser ver mAP por epoch — só ocupa mais
tempo.)

## 7. O que esperar nas comparações

- **Epoch 1 (`_epoch=0`)**: `epoch_open=False`. Caminho idêntico ao da
  Step 1. **`loss_oais` deve estar AUSENTE** do log (chave não criada).
  `loss_bbox`, `loss_cls`, `loss_rpn_*` próximos do `step3_ref` (dentro
  do ruído CUDA ~0.1–0.5%).
- **Epoch 2 em diante (`_epoch>=1`)**: `epoch_open=True`. Ambos ativos.
  - **`loss_oais` aparece** no log. Cálculo da expectativa, calibrado
    pela Step 2:

    ```
    loss_cls ~0.30 (epoch 3 baseline) ⇒ confiança média p ≈ exp(-0.30) ≈ 0.74
    Após max-pooling por bag (que pega o maior dentre vários RoIs/bag),
    inst_score ≈ 0.80–0.90.
    loss_oais ≈ (1 - 0.85) * 0.1 ≈ 0.015
    ```

  - `loss_bbox` continua divergente do baseline como na Step 2.
  - `loss_cls`, `loss_rpn_*` ainda próximos do baseline (OA-IS não toca
    cls; `loss_oais` empurra cls_score pra cima mas indiretamente —
    deve afetar pouco o `loss_cls` no curto prazo).
- **`loss` (total)** = `loss_rpn_cls + loss_rpn_bbox + loss_cls + loss_bbox +
  loss_oais` (a partir da epoch 2). O salto na epoch 1 → 2 é pequeno
  (~0.01–0.03 absoluto).

### Curva esperada de `loss_oais`

| Fase | Valor típico |
|---|---|
| Epoch 2 abre | ~0.01–0.03 |
| Epoch 2 fim | caindo lentamente |
| Epoch 3 fim | ~0.005–0.01 |

Esses números são pequenos em magnitude absoluta (já com
`oamil_lambda=0.1` aplicado). O sinal de saúde **não é** o valor
absoluto baixo — é a **tendência decrescente**.

Se ficar travado em ~0.015–0.03 e **não cair** ao longo das epochs,
algo bloqueia o gradiente (verificar que `confidence_scores` mantém
grad — checar o sanity log da seção 5 abaixo).

Se subir ou explodir: NaN, `oamil_lambda` grande demais, ou problema
nos scores.

## 8. Sinais de problema

| Sintoma | Provável causa |
|---|---|
| `NotImplementedError: OA-IE iteration` | Alguém setou `oaie_flag=True`; reverter |
| `RuntimeError: loss_oais became non-finite` | Confidence score 0 + grad NaN; investigar segundo forward |
| `loss_oais` constante (não decresce) | Grad bloqueado em algum `.detach()` indevido no caminho; verificar `_aggregate_instance_scores` e `_oamil_instance_selection` linhas D |
| `loss_oais` aparece na epoch 1 | `oamil_active` calculado errado — `_epoch` injetor pode não ter rodado, ou config com `oais_epoch=1` |
| `loss_oais` muito alto (>0.5) | `oamil_lambda` errado, ou cls_score muito negativo (não deveria, é softmax) |
| `loss_cls` explode na epoch 2 | Pseudo-target ruim influenciando convergência de cls — improvável mas reduzir `oais_theta` ajuda |
| Treino fica 2x mais lento na epoch 2+ | Esperado — segundo forward por iter + backward extra. ~40–80% overhead típico |
| `AssertionError: Step 3 must produce exactly one inst_score` | Bug futuro de Step 4 que não foi rolando direto. Nunca deve disparar na Step 3 |

## 9. Próximo passo

Quando passar a validação:

- Step 4: OA-IE (iteração múltipla). Vai ser o maior refactor — o
  segundo forward vira loop, `inst_scores_list` cresce, e a fórmula
  do `loss_oais` ramifica em `refine` vs `random`.

Por enquanto: **aguardando seus runs**. Não rodei treino.

---

## Apêndice — sanity check da implementação

Recapitulando o fluxo end-to-end para um batch (epoch ≥ 2):

```
1. RPN propõe ~1000 rois por imagem.
2. Assigner + Sampler → ~512 rois (positivos + negativos) por imagem.
3. _bbox_forward(x, rois) → cls_score, bbox_pred (primeiro forward).
4. get_targets → (labels, label_weights, bbox_targets, bbox_weights).
5. _oamil dispatcher abre os gates:
   - epoch_open=True → segue.
6. _oamil_instance_selection:
   a. Mask positivos: ~128 por batch (pos_fraction=0.25 * 512).
   b. Decode noisy_gt e pred_boxes (absolute coords).
   c. Bags por noisy_gt.sum(dim=1) → ~10-30 bags por batch.
   d. _bbox_forward(x, new_rois)  ← SEGUNDO forward (~128 rois).
   e. confidence_scores = softmax(cls_score_2nd)[:, pos_label].
   f. compute_pseudo=True → per-bag detach + phi + pseudo_gt + encode.
7. _aggregate_instance_scores:
   - Per-bag max → ~20 inst_scores.
   - Per-class mean → ~5-10 cls means (depende das classes na batch).
   - Mean over classes → 1 scalar inst_score com grad.
8. loss_oais = (1 - inst_score) * 0.1.
9. bbox_head.loss(...pseudo_bbox_targets=pseudo...) → loss_cls, acc,
   loss_bbox (loss_bbox usando pseudo, loss_cls inalterado).
10. losses.update({'loss_oais': loss_oais}).
11. mmengine.BaseModel.parse_losses soma todas chaves contendo 'loss' →
    total = loss_rpn_cls + loss_rpn_bbox + loss_cls + loss_bbox + loss_oais.
```

Tudo rodando dentro de um único batch_step. Cada batch da epoch 2+
paga um segundo forward através de duas FCs do bbox_head + um
RoIAlign, com gradient flow do `loss_oais` retornando.
