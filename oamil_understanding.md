# OA-MIL Port — Resumo de Entendimento e Mapeamento de API

Documento gerado antes da Step 1 do port. Serve para revisão fora do terminal.
Baseado em:

- `oamil_port_design.md` (design original)
- `/home/pesquisador/pesquisa/filipe/refs/oamil_ref/mmdet/models/roi_heads/standard_roi_head_oamil.py`
- `/home/pesquisador/pesquisa/filipe/refs/oamil_ref/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head_oamil.py`
- `/home/pesquisador/pesquisa/filipe/refs/oamil_ref/configs/_base_/models/faster_rcnn_r50_fpn_voc_oamil.py`
- `mmdet/models/roi_heads/standard_roi_head.py` (vcnc_mil, 3.x)
- `mmdet/models/roi_heads/bbox_heads/{bbox_head.py,convfc_bbox_head.py}` (vcnc_mil, 3.x)

---

## 1. Caminho dos refs (correção)

O design doc aponta para `~/pesquisa_oamil/refs/oamil_ref/`, mas os arquivos
estão de fato em `/home/pesquisador/pesquisa/filipe/refs/oamil_ref/`. Foi
desse caminho que li os 3 arquivos de referência.

## 2. Algoritmo (do que li nos refs, confirmando o pseudocódigo do design)

Dois "gates" temporais governados por `self._epoch + 1 >= oais_epoch / oaie_epoch`:

- **OA-IS** (instance selection): para cada GT ruidoso, agrupa os RoIs positivos
  que caíram nele (bag identificado por `torch.unique(noisy_gt_boxes.sum(dim=1))`),
  pega o de maior `softmax(cls_score)[:, gt_class]`, e mistura
  `pseudo_gt = phi · best_pred + (1 - phi) · noisy_gt`
  com `phi = clamp(score**gamma, max=theta)`. Encoda de volta pra delta-space e
  substitui `bbox_targets[pos_inds]` no cálculo de `loss_bbox`.
- **OA-IE** (instance extension): roda mais `oaie_num` iterações forward,
  alimentando as predições anteriores como novos rois. Coleta um score por
  iteração.
- **loss_oais**: agregação MIL.
  - `'refine'`: `(1 - s0) + (1 - mean(s1..N)) · oaie_coef`
  - `'random'`: `1 - mean(all_scores)`
  - multiplicada por `oamil_lambda`.
- Tudo pulado se `oamil_lambda == 0` (não há overhead nem alteração de loss).

## 3. Diferenças de API: mmdet 2.x (OA-MIL) → mmdet 3.x (vcnc_mil)

| Conceito | mmdet 2.x (ref) | mmdet 3.x (target) |
|---|---|---|
| Registro de model | `@HEADS.register_module()` | `@MODELS.register_module()` |
| Imports do registro | `from ..builder import HEADS, build_head, build_roi_extractor` | `from mmdet.registry import MODELS, TASK_UTILS` |
| `bbox2roi` | `from mmdet.core import bbox2roi` | `from mmdet.structures.bbox import bbox2roi` |
| Construir assigner/sampler | `build_assigner(cfg.assigner)`, `build_sampler(cfg.sampler, context=self)` | `TASK_UTILS.build(cfg.assigner)`, `TASK_UTILS.build(cfg.sampler, default_args=dict(context=self))` |
| Entry point de treino | `forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, ...)` | `loss(x, rpn_results_list, batch_data_samples)` — chama `unpack_gt_instances`, monta sampling_results, chama `bbox_loss` |
| Forward+target+loss do bbox head | sequência `_bbox_forward_train` (forward + `get_targets`) seguida de `bbox_head.loss(cls_score, bbox_pred, rois, *bbox_targets)` | método único `bbox_head.loss_and_target(cls_score, bbox_pred, rois, sampling_results, rcnn_train_cfg)` que internamente chama `get_targets` + `loss` |
| `get_targets` no bbox_head | `get_targets(sampling_results, gt_bboxes, gt_labels, train_cfg)` | `get_targets(sampling_results, rcnn_train_cfg)` (gt já mora em sampling_results) |
| `_get_targets_single` | em 2.x o ref passa `pos_gt_inds` extra (não usado dentro) | `(pos_priors, neg_priors, pos_gt_bboxes, pos_gt_labels, cfg)` — sem `pos_gt_inds` |
| `SamplingResult.pos_bboxes` | atributo direto | depreciado; renomeado para `pos_priors` (`pos_bboxes` ainda existe como property que warna) |
| RPN results | `proposal_list[i]` (Tensor) | `rpn_results_list[i]` (`InstanceData`); o `loss()` padrão renomeia `.bboxes → .priors` antes de chamar o assigner |
| Test entry | `simple_test` / `aug_test` | `predict_bbox` / `predict_by_feat` |
| Inicialização de pesos | `init_weights(pretrained)` manual | `init_cfg` declarativo via `BaseModule` |
| `bbox_coder.encode/decode` | `encode(bboxes, gt_bboxes)`, `decode(bboxes, pred_bboxes)` | idêntico — sem mudança de assinatura útil aqui |
| Hooks | `from mmcv.runner import Hook` | `from mmengine.hooks import Hook` (`from mmdet.registry import HOOKS`) |
| Epoch no runner | `runner.epoch` | `runner.epoch` (sem mudança, 0-indexed) |

## 4. Implicações concretas para o port

1. **Não precisa subclassar `ConvFCBBoxHead` inteiro.** O ref reimplementa tudo
   porque em 2.x ele precisou (faltava `init_cfg`, `MODELS.build(predictor_cfg_)`,
   etc.). Em 3.x basta `class Shared2FCBBoxHeadOAMIL(Shared2FCBBoxHead)`,
   sobrescrevendo apenas:
   - `__init__`: extrair kwargs OA-MIL antes de `super().__init__` e armazenar
     em `self`; inicializar `self._epoch = 0`.
   - `loss`: aceitar `pseudo_bbox_targets` opcional e usar no lugar de
     `bbox_targets[pos_inds]` quando passado.
   - `loss_and_target`: encaminhar `pseudo_bbox_targets` para `loss`.

2. **Não precisa subclassar `BaseRoIHead`.** Subclassar `StandardRoIHead` e
   sobrescrever **apenas `bbox_loss`** (não `loss`, não `_bbox_forward`). O
   `bbox_loss` 3.x é o equivalente do `_bbox_forward_train` 2.x.

3. **Fluxo proposto dentro do `bbox_loss` override**:
   1. `rois = bbox2roi([res.priors for res in sampling_results])`
   2. `bbox_results = self._bbox_forward(x, rois)`
   3. `cls_reg_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)`
      ← chamada explícita para ter `labels`/`bbox_targets` antes do OA-MIL
   4. `loss_oais, pseudo_bbox_targets = self._oamil(bbox_results, rois, cls_reg_targets, x)`
      (gate em `_epoch` e `oamil_lambda`)
   5. `losses = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *cls_reg_targets, pseudo_bbox_targets=pseudo_bbox_targets)`
   6. `losses.update(loss_oais)`
   7. retornar `dict(cls_score=..., bbox_pred=..., bbox_feats=..., loss_bbox=losses)`
      (formato esperado pelo `loss()` da `StandardRoIHead`)

4. **`SamplingResult.pos_priors`** é o equivalente de `pos_bboxes`. O ref usa
   `res.pos_bboxes` em `_mask_forward_train` — não vamos portar mask.

5. **Hook do `_epoch`**: idêntico ao planejado no design.
   ```
   runner.model.roi_head.bbox_head._epoch = runner.epoch
   ```
   em `before_train_epoch`. `runner.epoch` é 0-indexed; a comparação
   `_epoch + 1 >= oais_epoch` mantém a semântica do ref (1-indexed).

6. **Custom imports**: o repo já tem `custom_configs/hooks/` e
   `custom_configs/_base_/models/`. O design pede `custom_configs/models/`
   (novo) — confirmei que não existe ainda. Configs existentes usam o padrão
   `custom_imports=dict(imports=[...], allow_failed_imports=False)`.

7. **`reg_class_agnostic`**: o ref assume `False` (loss usa
   `bbox_pred.view(N, -1, 4)[pos_inds, labels[pos_inds]]`). Já é o padrão no
   `faster-rcnn_r50_fpn_20c.py` deste repo. OK.

## 5. Pontos de atenção / riscos

- `bbox_loss` deve manter a chave `bbox_feats` no dict retornado — senão o
  `mask_loss` (que existe na `StandardRoIHead.loss`) quebra, mesmo que não
  usemos mask em FasterRCNN. O `StandardRoIHead.loss` lê `bbox_results['bbox_feats']`.
- O `loss_and_target` padrão retorna `dict(loss_bbox=losses_dict, bbox_targets=cls_reg_targets)`
  — o `bbox_targets` aí é consumido por Cascade R-CNN. Não usamos Cascade, mas
  manter o formato consistente caso alguém estenda.
- A chamada extra a `get_targets` (para ter os targets antes do OA-MIL) tem
  custo trivial e é determinística; não tem efeito colateral em
  `sampling_results`.
- `_epoch` inicializado em 0 no `__init__` do bbox_head garante que mesmo sem
  o hook registrado, o gate `_epoch + 1 >= oais_epoch` ainda funciona como
  no ref (mas tecnicamente trava no epoch 0; o hook é necessário para o
  comportamento correto).

## 6. Arquivos a criar (Step 1, skeleton)

1. `custom_configs/models/__init__.py`
2. `custom_configs/models/shared2fc_bbox_head_oamil.py`
3. `custom_configs/models/standard_roi_head_oamil.py`
4. `custom_configs/hooks/oamil_epoch_injector_hook.py`
5. `custom_configs/exp_configs/exp_oamil_voc_sim40.py` (com `oamil_lambda=0` para
   validar que a loss bate com o baseline)

## 7. Decisões explícitas (resposta às 3 perguntas)

### (a) Onde `pseudo_bbox_targets` é computado e como flui

**Computado:** dentro de `StandardRoIHeadOAMIL._oamil_instance_selection`
(método novo no roi_head), seguindo as linhas 213–304 do ref 2.x.

**Fluxo (todo dentro do `bbox_loss` override do roi_head):**

```
StandardRoIHeadOAMIL.bbox_loss(x, sampling_results):
    1. rois = bbox2roi([res.priors for res in sampling_results])
    2. bbox_results = self._bbox_forward(x, rois)
        # bbox_results = {'cls_score', 'bbox_pred', 'bbox_feats'}
    3. cls_reg_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)
        # cls_reg_targets = (labels, label_weights, bbox_targets, bbox_weights)
    4. loss_oais_dict, pseudo_bbox_targets = self._oamil(
           bbox_results, rois, cls_reg_targets, x)
        # _oamil retorna ({}, None) se oamil_lambda == 0 OU epoch ainda não atingiu oais_epoch
        # caso contrário retorna ({'loss_oais': tensor}, pseudo_bbox_targets shape (num_pos, 4))
    5. losses = self.bbox_head.loss(
           bbox_results['cls_score'],
           bbox_results['bbox_pred'],
           rois,
           *cls_reg_targets,
           pseudo_bbox_targets=pseudo_bbox_targets)
        # losses = {'loss_cls', 'acc', 'loss_bbox'}
    6. losses.update(loss_oais_dict)
        # losses agora tem {'loss_cls', 'acc', 'loss_bbox', 'loss_oais'?}
    7. bbox_results.update(loss_bbox=losses)
    8. return bbox_results
```

**Contrato do kwarg `pseudo_bbox_targets` em `Shared2FCBBoxHeadOAMIL.loss`:**

- Tipo: `Optional[Tensor]`, default `None`.
- Shape quando passado: `(num_pos, 4)` — já indexado por `pos_inds`, conforme
  o ref 2.x (linhas 277–299 do ref roi_head e 311–325 do ref bbox_head).
- Quando `None`: comportamento idêntico ao baseline (usa `bbox_targets[pos_inds]`).
- Quando passado: substitui `bbox_targets[pos_inds]` no `self.loss_bbox(...)`
  da regressão. `bbox_weights[pos_inds]` continua sendo usado normalmente.

**Por que não passa via `loss_and_target`:** o roi_head chama `bbox_head.loss`
**diretamente** (skipping `loss_and_target`), porque precisamos dos `cls_reg_targets`
no meio do fluxo (entre forward e loss) para alimentar o OA-MIL. Reaproveitar
`loss_and_target` exigiria chamar `get_targets` duas vezes ou adicionar mais
um kwarg lá. Mais limpo desviar dele e replicar suas 2 linhas internas.

### (b) Como `_epoch` é injetado no bbox_head

**Quem injeta:** o hook `OAMILEpochInjectorHook` (arquivo 4 da seção 6) é
**criado por nós** como parte do port. Não assume injeção externa.

**Onde mora o atributo:**

- `Shared2FCBBoxHeadOAMIL.__init__` inicializa `self._epoch = 0` (fallback
  seguro, mas a comparação `_epoch + 1 >= oais_epoch` com `oais_epoch=2`
  garante que sem o hook OA-IS nunca ativa — comportamento conservador).
- O hook atualiza `runner.model.roi_head.bbox_head._epoch = runner.epoch`
  em `before_train_epoch` (0-indexed, mesmo padrão do mmengine).

**Como é lido:** dentro do roi_head (em `_oamil_instance_selection` e
no gate do OA-IE), via `self.bbox_head._epoch`. Não duplicamos o atributo
no roi_head — fica num único local (bbox_head) para evitar dessincronia.

**Por que no bbox_head e não no roi_head:** seguindo o design doc. O ref 2.x
deixava em `self._epoch` no roi_head, mas como `oais_epoch`/`oaie_epoch` já
são atributos do bbox_head, ter `_epoch` lá agrupa toda a configuração
temporal num só objeto. Custo do acesso `self.bbox_head._epoch` no roi_head
é zero.

**Wiring no config:** o hook precisa de:

```python
custom_imports = dict(imports=[
    'custom_configs.models.standard_roi_head_oamil',
    'custom_configs.models.shared2fc_bbox_head_oamil',
    'custom_configs.hooks.oamil_epoch_injector_hook',
], allow_failed_imports=False)

custom_hooks = [dict(type='OAMILEpochInjectorHook')]
```

Sem o `custom_hooks`, o `_epoch` fica congelado em 0 e OA-IS/OA-IE nunca
ativam (com os defaults `oais_epoch=2`, `oaie_epoch=9`).

### (c) Em que chave `loss_oais` é retornado, e quem soma com `loss_cls`/`loss_bbox`

**Chave:** `loss_oais` dentro do mesmo dict de losses que carrega `loss_cls`
e `loss_bbox`. Ou seja, ao final do `bbox_loss`, `bbox_results['loss_bbox']`
(o dict, não o tensor) é:

```python
{
    'loss_cls':  tensor(...),   # CE da classificação
    'acc':       tensor(...),   # accuracy, não somada (não tem 'loss' no nome)
    'loss_bbox': tensor(...),   # L1/SmoothL1 da regressão (usando pseudo se ativo)
    'loss_oais': tensor(...),   # presente só se oamil_lambda > 0 E gate OA-IS ativo
}
```

**Quem soma:** `mmengine.model.BaseModel.parse_losses` (verifiquei o source).
A regra é literal: `loss = sum(value for key, value in log_vars if 'loss' in key)`.
Qualquer chave contendo a substring `'loss'` entra na loss total automaticamente.

**Caminho completo:**

```
StandardRoIHeadOAMIL.bbox_loss   → retorna bbox_results com 'loss_bbox' = dict
StandardRoIHead.loss             → losses.update(bbox_results['loss_bbox'])
                                   # losses agora tem loss_cls, loss_bbox, loss_oais, acc
FasterRCNN (TwoStageDetector).loss → retorna losses dict combinado RPN+RoI
BaseModel.parse_losses           → soma todas as chaves com 'loss' no nome
                                   # loss_total = loss_rpn_cls + loss_rpn_bbox
                                   #            + loss_cls + loss_bbox + loss_oais
```

**Escala/peso:** o fator `oamil_lambda` já está multiplicado **dentro** do
`_compute_loss_oais` (lá no roi_head), seguindo o ref 2.x linha 207. Não há
peso adicional no config — o `oamil_lambda` no `bbox_head` é a única dial.

**Quando ausente:** se `oamil_lambda == 0` ou `epoch + 1 < oais_epoch`, a
chave `loss_oais` simplesmente **não é adicionada** ao dict. Não é zero — não
existe. Isso evita poluir o log com uma série de zeros e garante
equivalência numérica exata com o baseline na Step 1.

## 8. Arquivos a criar (Step 1, skeleton)

1. `custom_configs/models/__init__.py`
2. `custom_configs/models/shared2fc_bbox_head_oamil.py`
3. `custom_configs/models/standard_roi_head_oamil.py`
4. `custom_configs/hooks/oamil_epoch_injector_hook.py`
5. `custom_configs/exp_configs/exp_oamil_voc_sim40.py` (com `oamil_lambda=0`
   para validar que a loss bate com o baseline)
