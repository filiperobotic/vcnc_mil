# Diagnóstico — coleta vazia com `CocoDataset`

## 1. Onde fica a coleta

`custom_configs/hooks/vcnc_kmeans_confusion_aware_spatial_hook.py`:

- `before_train_epoch()` — linhas **555–691** (loop de coleta em 605–684, mensagem de erro em 686–688)
- `_build_image_map()` — linhas **1054–1060** (é aqui que está o problema)
- `_apply_relabel()` / `_apply_ignore_flag()` — linhas **1062–1076**
- `unwrap_to_leaf_datasets()` / `reload_leaf_datasets()` — linhas **39–76**

## 2. O que o hook assume do dataset

Ao contrário do que o nome sugere, o hook **não** depende de `ann_subdir`, XML, nem de nada específico do VOC. A única suposição estrutural é:

```python
# linha 1054-1060
def _build_image_map(self, datasets):
    img_map = {}
    for sub_idx, subds in enumerate(datasets):
        if hasattr(subds, 'data_list'):                       # <-- (a)
            for data_idx, data_info in enumerate(subds.data_list):   # <-- (b)
                img_map[data_info['img_path']] = (sub_idx, data_idx)
    return img_map
```

Ou seja: **`dataset.data_list` tem que estar populado em memória, como lista Python viva**, e as escritas de relabel (`datasets[sub_idx].data_list[data_idx]['instances'][gt_idx]`, linhas 1064 e 1072) precisam ser lidas de volta pelo dataloader.

`CocoDataset` produz exatamente os mesmos campos que `VOCDataset` (`img_path`, `instances[i]['bbox_label']`, `instances[i]['ignore_flag']` — `mmdet/datasets/coco.py`, `parse_data_info`). O formato **não** é a causa.

## 3. Causa raiz: `serialize_data`

O `BaseDataset` do mmengine (v0.10.7 instalado) **apaga `data_list` depois de serializar**:

`mmengine/dataset/base_dataset.py`:
```python
223:  serialize_data: bool = True,          # DEFAULT = True
...
306:  if self.serialize_data:
307:      self.data_bytes, self.data_address = self._serialize_data()
...
745:  def _serialize_data(self):
764:      data_list = [_serialize(x) for x in self.data_list]
...
771:      self.data_list.clear()               # <<<<<< data_list vira []
```

E a leitura passa a vir dos bytes, não da lista:
```python
260:  if self.serialize_data:
261-265:  ... pickle.loads(self.data_bytes[start:end])   # não olha data_list
267:  else: data_info = copy.deepcopy(self.data_list[idx])
```

## 4. Por que VOC funciona e COCO não

Os configs VOC do Daniel/Filipe passam explicitamente a flag — ex. `custom_configs/_base_/datasets/voc0712_daniel_loc20.py:105`:

```python
dataset=dict(
    type=dataset_type,
    ...
    ann_subdir = ann_subdir,
    serialize_data=False,   # Define como False (Filipe)   <-- linha 105
    ...
)
```

Todos os `voc0712_daniel_*.py` têm essa linha. **Nenhum** dos 15 `coco_detection*.py` tem — confirmei com `grep -L`:

```
custom_configs/_base_/datasets/coco_detection_loc20.py   (o usado pelo config em questão)
custom_configs/_base_/datasets/coco_detection.py
custom_configs/_base_/datasets/coco_detection_loc40.py
... (todos os 15)
```

No `coco_detection_loc20.py:44-52`, o bloco `dataset=dict(...)` vai direto de `filter_cfg` para `pipeline`, sem `serialize_data`.

## 5. A cadeia exata da falha

1. Config COCO → `serialize_data=True` (default) → no `full_init()` inicial, `data_list.clear()` roda (linha 771).
2. `reload_dataset=True` (linha 69 do config) chama `reload_leaf_datasets()` → força `_fully_initialized=False` + `full_init()` → **serializa e limpa `data_list` de novo**, toda época.
3. `_build_image_map()` (linha 584): `hasattr(subds, 'data_list')` é `True` (o atributo existe), mas a lista está **vazia** → `dataset_img_map == {}`.
4. Loop de coleta, linhas 615–616:
   ```python
   if img_path not in dataset_img_map:
       continue
   ```
   → `continue` para **todas** as imagens de **todos** os batches.
5. `all_box_data` fica vazio → linha 686 → `"[VCNC-Spatial] Nenhum box coletado!"` → `return`, em toda época pós-warmup.

O custo é agravado: o loop das linhas 605–610 ainda roda o forward completo (`my_get_logits`) sobre o train set inteiro antes de descartar tudo — você paga a inferência de uma época extra por época, para nada.

## 6. Bug secundário (não causa o vazio, mas quebra a correção)

Dois pontos que só aparecerão *depois* de resolver o item 5:

- **Escrita ignorada:** mesmo com o mapa populado, com `serialize_data=True` as escritas em `data_list` (linhas 1065 e 1073) nunca seriam lidas, porque `get_data_info` lê de `data_bytes` (linhas 260–265). Ou seja, `serialize_data=False` não é só conveniência — é requisito do design de relabel in-place.

- **Desalinhamento de `gt_idx` com `iscrowd`:** o loop usa `gt_idx` de `assign_result` (linha 638), que indexa `data_sample.gt_instances`. Em `PackDetInputs` (`mmdet/datasets/transforms/formatting.py:88-90`), instâncias com `ignore_flag == 1` são removidas para `ignored_instances`. Então `gt_idx` indexa a lista **filtrada**, enquanto `_apply_relabel` usa esse mesmo índice contra a lista **completa** `data_list[data_idx]['instances']` (linha 1064). Em COCO, `iscrowd` → `ignore_flag = 1` (`coco.py`, `parse_data_info`); em VOC, `difficult` ou `bbox_min_size` → `ignore_flag = 1` (`xml_style.py:154-157`).

  **Quanto isso pega, na prática (medido nos dados atuais):** quase nada — os geradores de ruído zeram os dois campos.

  | conjunto de treino | instâncias com `ignore_flag = 1` |
  |---|---|
  | COCO `annotations/instances_train2017.json` (baseline limpo, usado só por `coco_detection.py`) | **10.052** `iscrowd=1` / 860.001 |
  | COCO, os 25 `NoiseAnnotations/instances_train2017_*.json` | 0 / 860.001 (e demais totais) |
  | VOC, os 25 `ann_subdir` de treino dos `voc0712_daniel_*.py` | 0 arquivos com `<difficult>1</difficult>` / 16.551 XMLs cada |
  | VOC `original/Annotations` (só validação — o hook não toca) | 4.804 / 21.503 arquivos com `difficult=1` |

  Nos configs VOC o `filter_cfg` está comentado (ex. `voc0712_daniel_loc20.py:106-107`), logo `bbox_min_size = None` e `ignore` é sempre `False`. Somando com o `difficult=0` universal nos XMLs de ruído: em VOC vale `ignore_flag ≡ 0`, ou seja `orig_idx == gt_idx` e **nunca houve relabel em índice errado** — os resultados VOC antigos não são afetados. O mesmo vale para os 14 configs COCO de ruído. O desalinhamento só é materialmente alcançável no `coco_detection.py` (baseline COCO limpo). Corrigir continua valendo a pena como blindagem, caso voltem a usar anotações que preservem `iscrowd`/`difficult`.

## 7. Status

Ambos os fixes foram aplicados (autorizados em 2026-08-18):

- **FIX A:** `serialize_data=False` no `dataset` do `train_dataloader` dos 15 `custom_configs/_base_/datasets/coco_detection*.py` (val/test intocados).
- **FIX B:** novo `_resolve_orig_instance_idx()` no hook, usado por `_apply_relabel`/`_apply_ignore_flag`, com warning + skip quando `gt_idx` sai do intervalo. Inclui um cache por época (`self._valid_idx_cache`, zerado em `before_train_epoch`) para impedir que um `ignore_flag = 1` escrito na Etapa 1/2 desloque os índices resolvidos nas Etapas 2/3/4 da mesma imagem.

Detalhamento em `diag2_coco.md`.
