# Step 2 — Validation Plan

OA-IS implementado. OA-IE e `loss_oais` continuam stubados.

## 1. Arquivos modificados

| Arquivo | O que mudou |
|---|---|
| `custom_configs/models/shared2fc_bbox_head_oamil.py` | `loss()` agora consome `pseudo_bbox_targets` via clone-and-replace, depois delega ao `super().loss`. Path de `cls_score` continua bit-idêntico ao baseline. Assert de shape. |
| `custom_configs/models/standard_roi_head_oamil.py` | Novo método `_oamil_instance_selection` com toda a lógica OA-IS. `_oamil` agora roteia: OA-IE → `NotImplementedError`, `oamil_lambda > 0` → `NotImplementedError`, OA-IS ativo → executa; senão `({}, None)`. |
| `custom_configs/exp_configs/exp_oamil_voc_sim40.py` | `oais_flag=True`. Outros campos OA-MIL inalterados. |

## 2. Onde a lógica de bag identification roda

Arquivo: `custom_configs/models/standard_roi_head_oamil.py`,
método `_oamil_instance_selection`. Mapeando linha a linha vs o ref OA-MIL
(`refs/oamil_ref/.../standard_roi_head_oamil.py`):

| Etapa | Step 2 (nosso) | Ref OA-MIL 2.x |
|---|---|---|
| Mask de positivos | linhas 122–125 | linhas 225, 230 |
| Decode noisy_gt | linhas 134–135 | linhas 228 |
| Decode predictions | linhas 136–137 | linhas 256 (dentro do loop OA-IE) |
| Bag id por `noisy_gt.sum(dim=1)` | linhas 140–146 | linhas 229–230 |
| Segundo forward em `pred_boxes` | linhas 149–157 | linhas 260, 266 |
| `softmax(cls_score)[:, gt_class]` | linhas 153–157 | linha 267 |
| `phi = clamp(score^gamma, max=theta)` | linha 166 | linha 294 |
| `pseudo_gt = phi·best + (1-phi)·noisy` | linhas 169–170 | linha 295 |
| Encode pseudo_gt → delta | linhas 172–175 | linha 298 |

Identificação de bag: `torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True,
return_inverse=True)`. RoIs amostradas que caem no mesmo GT ruidoso têm o
mesmo `pos_assigned_gt_inds` no sampler, portanto o mesmo
`pos_gt_bboxes` (mesmo box), portanto a mesma soma. Hash determinístico
por construção. Colisão entre GTs distintos exigiria dois boxes com
exatamente a mesma soma `x1+y1+x2+y2` — astronomicamente improvável para
coordenadas float arbitrárias, e mesmo se acontecer o efeito é mesclar
duas bags em uma (degradação leve, não crash).

## 3. Como o `pseudo_gt` é computado (por bag)

```python
bag_scores       = confidence_scores[in_bag]            # (B,)
bag_pred_boxes   = pred_boxes[in_bag]                   # (B, 4) absolute
best_pos         = bag_scores.argmax()                  # escalar
best_score       = bag_scores[best_pos].detach()        # escalar
best_pred_box    = bag_pred_boxes[best_pos].detach()    # (1, 4)

phi              = clamp(best_score ** oais_gamma, max=oais_theta)
noisy_gt_this    = noisy_gt_boxes[in_bag[0]]            # (1, 4) — qualquer
                                                         # row do bag serve,
                                                         # todas têm o mesmo
                                                         # noisy GT
pseudo_gt        = phi · best_pred_box + (1 − phi) · noisy_gt_this   # (1, 4)
pseudo_targets   = bbox_coder.encode(rois[in_bag],
                                     pseudo_gt.repeat(B, 1))         # (B, 4)
```

Detalhes importantes:

- `best_score.detach()` e `best_pred_box.detach()` — `phi` e `pseudo_gt`
  são tratados como constantes pelo loss. Sem fluxo de gradiente vindo
  da seleção de instância (Step 2 ignora isso; Step 3 vai usar
  `confidence_scores` sem detach pro `loss_oais`).
- `pseudo_gt` é **absoluto** (xyxy); só depois é encodado pra delta-space
  via `bbox_coder.encode` contra as **RoIs originais** (não contra os
  predicted boxes). Isso casa com o sistema de coordenadas que o
  `bbox_head.loss` espera em `bbox_targets`.
- `pseudo_gt.repeat(B, 1)` — todas as posições do bag recebem o MESMO
  pseudo_gt absoluto, mas o `encode` produz deltas DIFERENTES porque as
  RoIs de partida são diferentes. É exatamente o que o ref faz
  (linha 298).
- `phi ≤ theta = 0.85`, então o pseudo_gt nunca puxa mais que 85% pra
  cima do best_pred. Com `gamma=7.5` e best_score baixo (~0.3), phi
  fica em ~0.0001 — pseudo_gt ≈ noisy_gt. Só quando o modelo está bem
  confiante (best_score próximo de 1) é que phi se aproxima de theta.

## 4. Asserts e checagens defensivas

Em `Shared2FCBBoxHeadOAMIL.loss` (quando `pseudo_bbox_targets` chega):

```python
assert pseudo_bbox_targets.shape == bbox_targets[pos_mask].shape
```

Pega bug do roi_head produzindo targets do tamanho errado (ex.: sem
filtrar positivos antes).

Em `_oamil_instance_selection`:

```python
assert not bbox_head.reg_class_agnostic, '...'
```

Step 2 assume regressão por classe — o slice
`bbox_pred.view(N, -1, 4)[pos_inds, pos_labels]` só faz sentido nesse
caso. Se alguém ligar `reg_class_agnostic=True` num config futuro,
falha cedo com mensagem clara em vez de produzir lixo silencioso.

```python
assert not bbox_head.reg_decoded_bbox, '...'
```

Step 2 produz `pseudo_bbox_targets` em **delta-space** (via
`bbox_coder.encode`). Se o head estiver configurado pra targets em
coords absolutas (`reg_decoded_bbox=True`, usado por GIoULoss/IoULoss),
o swap silencioso colocaria targets na escala errada e a regressão
explode. Falha cedo.

```python
assert torch.isfinite(pseudo_bbox_targets).all()
```

Cinto-de-segurança contra NaN/Inf emergindo do `bbox_coder.encode` em
boxes degenerados (largura/altura zero). Caso aconteça, a mensagem
aponta diretamente pros campos a verificar (`oais_gamma`, `oais_theta`,
boxes preditos).

```python
if not pos_mask.any():
    return None
```

Imagem sem positivos no batch — retorna `None`, `bbox_head.loss` cai
no caminho baseline. Sem crash.

## 5. Gates de ativação

Resumo da máquina de estados em `_oamil`:

```
oaie_active  = oaie_flag AND _epoch + 1 >= oaie_epoch
  → NotImplementedError (Step 4)

oamil_lambda > 0
  → NotImplementedError (Step 3)

oais_active  = oais_flag AND _epoch + 1 >= oais_epoch
  → roda _oamil_instance_selection

senão
  → ({}, None)  # comportamento da Step 1
```

Com o config atual (`oais_flag=True, oais_epoch=2, oaie_flag=False,
oamil_lambda=0`):

- `_epoch=0` (epoch 1 humana): `0+1=1 >= 2` → False → `pseudo=None`,
  comportamento Step 1.
- `_epoch=1` (epoch 2 humana): `1+1=2 >= 2` → True → OA-IS roda.
- Idem epochs subsequentes.

## 6. Comandos sugeridos pra você validar

3 epochs deve ser suficiente pra ver tanto o regime "antes" (epoch 1)
quanto "depois" (epoch 2–3). Use o clean baseline diagnóstico como
referência pra epoch 1:

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil

# Run de referência (sem OA-MIL)
python tools/train.py \
    custom_configs/exp_configs/_diag_clean_baseline.py \
    --work-dir work_dirs/step2_ref \
    --cfg-options \
        randomness.deterministic=True \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=999 \
        default_hooks.logger.interval=50

# Run com OA-IS ativo
python tools/train.py \
    custom_configs/exp_configs/exp_oamil_voc_sim40.py \
    --work-dir work_dirs/step2_oais \
    --cfg-options \
        randomness.deterministic=True \
        train_cfg.max_epochs=3 \
        train_cfg.val_interval=999 \
        default_hooks.logger.interval=50
```

(`val_interval=999` pula validação intermediária e poupa ~50% do tempo
total; remova se quiser ver mAP por epoch.)

## 7. O que esperar nas comparações

- **Epoch 1 (`_epoch=0`)**: `oais_active=False`, então
  `pseudo_bbox_targets=None`. Caminho idêntico à Step 1. Loss valores
  devem ficar dentro do ruído CUDA vs `step2_ref` (~0.1–0.5% drift no
  iter-50, consistente com a investigação da Step 1).
- **Epoch 2 em diante (`_epoch>=1`)**: OA-IS ativo. `loss_bbox` deve
  **divergir** do baseline — esse é o ponto. `loss_cls` e `loss_rpn_*`
  devem permanecer próximas, porque OA-IS só toca o target de
  regressão. `acc` da classificação também deve ficar próxima.
- **`loss_oais` continua ausente** do log (oamil_lambda=0). Step 3 que
  vai trazer essa chave.
- **Sem crash, sem NaN, sem warning de assert**. Se algum assert
  disparar, o stacktrace já diz qual hipótese quebrou.

## 8. Sinais de problema

| Sintoma | Provável causa |
|---|---|
| `NotImplementedError: OA-IE iteration is not implemented yet` | Alguém setou `oaie_flag=True` no config; reverter |
| `NotImplementedError: loss_oais aggregation is not implemented yet` | `oamil_lambda > 0` no config; reverter pra 0 |
| `AssertionError: pseudo_bbox_targets shape mismatch` | Bug — `_oamil_instance_selection` retornou shape errado; verificar mask |
| `AssertionError: ... reg_class_agnostic=False` | Config com `reg_class_agnostic=True` (não suportado em Step 2) |
| `AssertionError: ... reg_decoded_bbox=False` | Loss type que usa coords absolutos (GIoU/IoU) — incompatível com encode delta |
| `AssertionError: pseudo_bbox_targets contains NaN/Inf` | Box predito degenerado (w=0 ou h=0); investigar `pred_boxes.clamp` ou `oais_theta` |
| `loss_bbox` **idêntico** à baseline em epoch 2 | OA-IS não disparou; checar `oais_flag=True` e que `OAMILEpochInjectorHook` está em `custom_hooks` (sem ele `_epoch` fica 0 pra sempre) |
| `loss_bbox` explode em epoch 2 | phi alto demais cedo demais; reduzir `oais_theta` ou aumentar `oais_gamma` |

## 9. Próximo passo

Aguardando você rodar e confirmar:

1. Epoch 1 ≈ baseline (ruído CUDA).
2. Epoch 2+ tem `loss_bbox` diferente do baseline (sem NaN, sem crash).
3. Nenhum assert disparado.

Se passar: Step 3 (adicionar `loss_oais` com `oamil_lambda=0.1`, mantendo
OA-IE off).

**Não rodei treino.** Modificações apenas no disco.
