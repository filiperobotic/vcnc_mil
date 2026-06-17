# Step 1 — Validation Plan

Esqueleto do port OA-MIL criado. Nenhuma lógica OA-IS/OA-IE ainda. Com
`oamil_lambda=0`, o caminho do código deve ser numericamente idêntico ao
baseline `StandardRoIHead` + `Shared2FCBBoxHead`.

## 1. Arquivos criados

Todos paths absolutos, sob `/home/pesquisador/pesquisa/filipe/vcnc_mil/`:

| # | Arquivo | Conteúdo |
|---|---|---|
| 1 | `custom_configs/models/__init__.py` | Reexporta `Shared2FCBBoxHeadOAMIL` e `StandardRoIHeadOAMIL` para o pacote |
| 2 | `custom_configs/models/shared2fc_bbox_head_oamil.py` | Subclasse de `Shared2FCBBoxHead`; guarda hyperparams OA-MIL e `_epoch=0`; `loss` aceita `pseudo_bbox_targets` (no skeleton, sempre `None` → delega a `super().loss`) |
| 3 | `custom_configs/models/standard_roi_head_oamil.py` | Subclasse de `StandardRoIHead`; sobrescreve `bbox_loss` para fluxo OA-MIL (com `_oamil` retornando `({}, None)` no skeleton); `loss_oais` **ausente** do dict quando inativo |
| 4 | `custom_configs/hooks/oamil_epoch_injector_hook.py` | `OAMILEpochInjectorHook.before_train_epoch` injeta `runner.epoch` em `roi_head.bbox_head._epoch` (desembrulha DDP, no-op se atributo ausente) |
| 5 | `custom_configs/exp_configs/exp_oamil_voc_sim40.py` | Config de teste: aponta para as novas classes, `oamil_lambda=0`, registra `OAMILEpochInjectorHook`, mesmo schedule do baseline sim40 |

## 2. Garantia de equivalência numérica (o que o skeleton faz)

Caminho do baseline (`StandardRoIHead.bbox_loss` → `bbox_head.loss_and_target`):

```
rois = bbox2roi([res.priors for res in sampling_results])
bbox_results = self._bbox_forward(x, rois)
cls_reg_targets = bbox_head.get_targets(sampling_results, train_cfg)  # interno ao loss_and_target
losses = bbox_head.loss(cls_score, bbox_pred, rois, *cls_reg_targets)  # interno ao loss_and_target
```

Caminho do skeleton (`StandardRoIHeadOAMIL.bbox_loss`):

```
rois = bbox2roi([res.priors for res in sampling_results])
bbox_results = self._bbox_forward(x, rois)
cls_reg_targets = bbox_head.get_targets(sampling_results, train_cfg)  # explícito
loss_oais_dict, pseudo_bbox_targets = self._oamil(...)  # → ({}, None) com lambda=0
losses = bbox_head.loss(cls_score, bbox_pred, rois, *cls_reg_targets, pseudo_bbox_targets=None)
losses.update({})  # no-op
```

Com `pseudo_bbox_targets=None`, `Shared2FCBBoxHeadOAMIL.loss` delega
exatamente a `super().loss(cls_score, bbox_pred, rois, *cls_reg_targets,
reduction_override=None)`. Mesmas chamadas, mesmos args, mesma matemática.

Chave `loss_oais` não é adicionada (dict update vazio), então
`mmengine.BaseModel.parse_losses` — que soma `'loss' in key` — produz
exatamente as mesmas chaves e valores que o baseline.

## 3. Como você testa a equivalência

Você precisa rodar duas vezes, mesma seed, e comparar as primeiras N
iterações. O config novo já fixa `randomness=dict(seed=2025)`; o baseline
não — então passe a seed pelo CLI para ambos.

**Baseline (já existe no repo):**

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil
python tools/train.py \
    'custom_configs/faster_rcnn/exp5_12x1[9]_baseline_danieldb_sim40.py' \
    --work-dir work_dirs/step1_baseline \
    --cfg-options randomness.seed=2025 train_cfg.max_epochs=1
```

**Skeleton OA-MIL (oamil_lambda=0):**

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil
python tools/train.py \
    custom_configs/exp_configs/exp_oamil_voc_sim40.py \
    --work-dir work_dirs/step1_oamil_lambda0 \
    --cfg-options train_cfg.max_epochs=1
```

(A seed do skeleton já está no config; o `--cfg-options` no baseline força a
mesma seed.)

### O que comparar

Em `work_dirs/step1_baseline/<timestamp>/vis_data/scalars.json` e
`work_dirs/step1_oamil_lambda0/<timestamp>/vis_data/scalars.json`, ou nos
logs `*.log`, observe **as primeiras ~10 iterações**. Devem bater:

- `loss_rpn_cls`
- `loss_rpn_bbox`
- `loss_cls`
- `loss_bbox`
- `loss` (soma total)
- `acc`

E **não** deve aparecer `loss_oais` no skeleton (chave ausente, não zero).

Critério de aceitação: diferença abaixo do ruído de ponto flutuante
(idealmente bit-exact nas primeiras iterações; pequena divergência
acumulada após várias iterações é aceitável e indica determinismo
imperfeito do CUDA, não bug do port).

## 4. Comando rápido para um único training step

Se você só quer ver 1 iteração rodar sem treinar 1 epoch inteira:

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil
python tools/train.py \
    custom_configs/exp_configs/exp_oamil_voc_sim40.py \
    --work-dir work_dirs/step1_smoke \
    --cfg-options \
        train_cfg.max_epochs=1 \
        default_hooks.logger.interval=1
```

Pare com Ctrl+C depois de algumas iterações. O log mostra `loss_cls`,
`loss_bbox`, etc. — e **não** deve mostrar `loss_oais`.

## 5. Se algo der errado — diagnóstico rápido

| Sintoma | Provável causa |
|---|---|
| `KeyError: 'StandardRoIHeadOAMIL'` | `custom_imports` não está sendo aplicado; verifique que o config tem `custom_imports = dict(...)` e que `tools/train.py` carrega o config certo |
| `NotImplementedError: OA-MIL logic is not implemented` | Algum config setou `oamil_lambda > 0` — não era pra acontecer na Step 1 |
| `NotImplementedError: pseudo_bbox_targets handling lands in Step 2` | Mesma coisa do anterior; o stub `_oamil` está sendo chamado com `oamil_lambda > 0` |
| `loss_oais` aparece no log | Bug — não deveria existir no skeleton; verifique se `bbox_head.oamil_lambda == 0` |
| Losses divergem entre baseline e skeleton | Verifique se a seed bate em ambos os runs e se `randomness.deterministic` está igual |
| `AttributeError: bbox_head has no attribute _epoch` (no hook) | O config não importou `shared2fc_bbox_head_oamil` antes do hook rodar; o hook é defensivo (`hasattr`), mas vale checar o `custom_imports` |

## 6. Próximos passos

Aguardando sua revisão. Se Step 1 passar, próxima é **Step 2** (OA-IS only,
lambda ainda 0, computa `pseudo_bbox_targets` mas não adiciona `loss_oais`).

---

**Não rodei o treino.** Os arquivos estão no disco prontos para você
executar manualmente.
