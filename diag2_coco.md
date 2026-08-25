Feito. Nada foi executado (só `py_compile` e leitura de arquivos de anotação para a análise do item 3).

## 1. Diff do FIX B

Diff completo mostrado acima. Resumo do que mudou em `custom_configs/hooks/vcnc_kmeans_confusion_aware_spatial_hook.py`:

- **Novo `_resolve_orig_instance_idx()`** (linha ~1072): constrói `valid_orig_idx` exatamente como você especificou, retorna `valid_orig_idx[gt_idx]`, e no caso `gt_idx >= len(valid_orig_idx)` emite `self._logger.warning(...)` com `img_path` + contagens e retorna `None` (o chamador dá `return`, sem `IndexError`).
- **`_apply_relabel` / `_apply_ignore_flag`**: passam a escrever em `instances[orig_idx]` e ganharam o kwarg `img_path=None`.
- **5 call sites** (E1 filter, E1 relabel, E2 filter, E2 relabel, E3 relabel, E4 filter) passam `img_path=box['img_path']`.

**Um desvio da sua especificação, que julguei necessário:** `_apply_ignore_flag` escreve `ignore_flag = 1` na mesma lista de onde `valid_orig_idx` é derivado. Como as etapas rodam em sequência sobre a mesma imagem, um filtro da E1/E2 encolheria a lista de válidos e todos os `gt_idx` resolvidos depois (E2/E3/E4) apontariam para a instância errada — ou cairiam no warning. Na sua config isso é imediatamente alcançável: E2 gate filter está ON e E3 spatial relabel está ON.

Exemplo concreto: imagem com instâncias `[A, B, C]`, todas válidas. E2 filtra `gt_idx=0` → `A.ignore_flag=1`. E3 quer relabelar `gt_idx=2` (C) → `valid_orig_idx` recalculado vira `[1, 2]` → `2 >= 2` → warning e skip.

A correção é memoizar `valid_orig_idx` por `(sub_idx, data_idx)` em `self._valid_idx_cache`, zerado no início de cada `before_train_epoch` (linha 586, logo após o reload). Como o cache é preenchido na primeira chamada para aquela imagem — sempre *antes* da primeira mutação nela —, ele congela o mapeamento no estado em que os `gt_idx` foram produzidos na coleta. A list comprehension que você escreveu está intacta; só ganhou o wrapper de cache.

## 2. FIX A — os 15 configs

Inserido `serialize_data=False,` no `dataset=dict(...)` do **`train_dataloader`**, entre `filter_cfg` e `pipeline=train_pipeline` (mesma posição relativa dos configs VOC). `val_dataloader`/`test_dataloader` intocados — verificado, 1 inserção por arquivo:

```
custom_configs/_base_/datasets/coco_detection.py
custom_configs/_base_/datasets/coco_detection_asim20.py
custom_configs/_base_/datasets/coco_detection_asim40.py
custom_configs/_base_/datasets/coco_detection_loc20.py
custom_configs/_base_/datasets/coco_detection_loc20_asim20.py
custom_configs/_base_/datasets/coco_detection_loc20_mi20_bkg20_asim20.py
custom_configs/_base_/datasets/coco_detection_loc20_mi20_bkg20_sim20.py
custom_configs/_base_/datasets/coco_detection_loc40.py
custom_configs/_base_/datasets/coco_detection_loc40_asim40.py
custom_configs/_base_/datasets/coco_detection_loc40_mi40_bkg40_asim40.py
custom_configs/_base_/datasets/coco_detection_loc40_mi40_bkg40_sim40.py
custom_configs/_base_/datasets/coco_detection_loc60.py
custom_configs/_base_/datasets/coco_detection_sim20.py
custom_configs/_base_/datasets/coco_detection_sim40.py
custom_configs/_base_/datasets/coco_detection_sim60.py
```

Todos os 15 e o hook passam em `py_compile`.

## 3. Impacto do FIX B nos experimentos VOC antigos: **nenhum — é no-op**

Fui checar em vez de deduzir, e o resultado contradiz o que eu afirmei no item 6 do diagnóstico. Correção:

Em VOC, `ignore_flag = 1` só é setado quando `difficult or ignore` (`mmdet/datasets/xml_style.py:154-157`), onde `ignore` depende de `bbox_min_size`:

- **`difficult`:** varri os 25 `ann_subdir` de treino distintos referenciados pelos `voc0712_daniel_*.py`. **Zero** arquivos com `<difficult>1</difficult>` em todos os 25 (16.551 XMLs cada). Os XMLs de ruído do Daniel são regerados com `<difficult>0</difficult>` em todos os objetos. Para comparação, o `original/Annotations` (usado só na validação, que o hook não toca) tem 4.804/21.503 arquivos com difficult=1 — ou seja, a informação existe no VOC original e foi zerada na geração do ruído.
- **`bbox_min_size`:** o `filter_cfg` está comentado nos configs de treino (ex. `voc0712_daniel_loc20.py:106-107`), então `bbox_min_size = None` e `ignore` é sempre `False`.

Logo, em todo experimento VOC de vocês `ignore_flag ≡ 0`, `valid_orig_idx == list(range(len(instances)))`, e `orig_idx == gt_idx`. **Nunca houve índice errado em VOC** — o relabel sempre bateu na instância certa. Os resultados VOC antigos não são invalidados e não precisam ser re-rodados.

E o mesmo vale para 14 dos 15 configs COCO: os arquivos de `NoiseAnnotations/` também zeram `iscrowd`. Contagem de `iscrowd=1` nos arquivos de treino:

| arquivo | iscrowd=1 / total |
|---|---|
| `annotations/instances_train2017.json` | **10.052** / 860.001 |
| todos os 25 `NoiseAnnotations/instances_train2017_*.json` | 0 |

Então o desalinhamento só é materialmente alcançável no `coco_detection.py` (baseline COCO limpo, sem ruído). Nos configs de ruído o FIX B também é no-op.

**A correção conserta a aplicação, não muda o método.** Não altera critério de relabel, gate de confusão, clustering nem spatial refinement — só garante que o índice escrito corresponda ao box avaliado. Com os dados atuais o comportamento numérico é idêntico ao anterior em VOC e nos 14 COCO ruidosos; o valor é blindar contra `iscrowd`/`difficult` caso vocês voltem a usar anotações que os preservem (e o warning te avisa se isso acontecer).

Uma observação sobre o baseline `coco_detection.py`: nele, as 10.052 anotações crowd continuam sendo excluídas do `gt_instances` pelo `PackDetInputs`, então o VCNC nunca as avalia — o FIX B apenas garante que os relabels dos boxes vizinhos não escorreguem por causa delas.

O item 6 do `diag_coleta_coco.md` afirma que COCO "tem muitas anotações crowd" e que o relabel "vai bater em instâncias erradas" — isso vale para o baseline limpo, mas não para os configs de ruído que vocês de fato rodam. Quer que eu atualize esse trecho do arquivo com os números acima?
