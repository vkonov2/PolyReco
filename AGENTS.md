# AGENTS.md

Инструкции для всех агентов, работающих в проекте `PolyReco`.

## Общие правила

- Всегда отвечать пользователю на русском языке.
- Перед изменениями читать существующий код и сохранять текущий стиль проекта.
- Не откатывать и не перезаписывать чужие изменения в git. Если рабочее дерево грязное, работать только с файлами, относящимися к задаче.
- Не создавать много разных файлов для визуализации. По возможности переиспользовать существующие HTML/JSON в `output/` или явно выбранные пользователем пути.
- Генерируемые артефакты держать в `output/`, если пользователь не указал другой путь.
- Не добавлять новые крупные зависимости без необходимости.
- Для поиска использовать `rg`, для списка файлов `rg --files`.

## Назначение проекта

Проект работает с полиэдральными 3D-моделями и теневыми контурами:

- читает `InitialModel` и файлы теневых контуров;
- строит проекции, сечения и линии по уровням `Z`;
- вычисляет функции расстояний/ошибок;
- сопоставляет наблюдаемые точки с ребрами эталонной модели;
- генерирует интерактивные HTML-визуализации.

Основные данные лежат в `data/<model>/`:

- `InitialModel` - исходная модель;
- `shadow/merged-cont*` - контуры;
- `reflect_*.obj` или `Reflect_*.obj` - дополнительные OBJ-модели, если есть.

Сейчас в `data/` есть модели: `round`, `princess`, `radiant`, `pear`, `cushion`.

## Окружение

Проект использует Python `>=3.13` и `uv`.

Установка/синхронизация зависимостей:

```bash
uv sync
```

Запуск скриптов предпочтительно делать так:

```bash
uv run python src/<script>.py ...
```

В `pyproject.toml` указаны зависимости:

- `numpy`
- `highspy`
- `matplotlib`
- `tqdm`

Тестового набора в репозитории сейчас нет. Для проверки использовать запуск нужного скрипта с малым `--max-contours`, если полный расчет долгий.

## Структура кода

- `src/polyreco/io.py` - парсинг моделей и контуров.
- `src/polyreco/model_data.py` - dataclass-структуры модели, вершин, граней, контуров.
- `src/polyreco/contour.py` - подготовка 2D-контуров и нормалей.
- `src/polyreco/lp_model.py` - LP-модель через HiGHS.
- `src/polyreco/solve.py` - оптимизационный solve-слой.
- `src/polyreco/export.py` - экспорт модели.
- `src/run_fit.py` - старый/минимальный запуск fit-пайплайна.
- `src/run_shadow_projection_viewer.py` - интерактивный matplotlib viewer модели и одного контура.
- `src/run_window_line_functions_viewer.py` - viewer функций расстояний по sliding window.
- `src/generate_combined_cached_viewer.py` - HTML viewer с объединенной проекцией и функциями.
- `src/generate_full_circle_split_cached_viewer.py` - основной HTML viewer full-circle split и RMS-минимумов.
- `src/match_full_circle_points_to_edges.py` - сопоставление точек с ребрами модели и HTML/JSON результат.

## Основные команды

Быстрый просмотр проекций модели и контуров:

```bash
uv run python src/run_shadow_projection_viewer.py --model-name round --max-contours 50
```

Просмотр функций расстояний по окнам:

```bash
uv run python src/run_window_line_functions_viewer.py --model-name round --z-step 0.01 --window-size 10 --max-contours 100
```

Генерация full-circle split viewer:

```bash
uv run python src/generate_full_circle_split_cached_viewer.py --models round --output-html output/full_circle_split_dynamic_viewer.html
```

Пересборка cache для full-circle split viewer:

```bash
uv run python src/generate_full_circle_split_cached_viewer.py --models round --rebuild-cache --cache-json output/full_circle_split_dynamic_models_cache.json --output-html output/full_circle_split_dynamic_viewer.html
```

Генерация combined viewer:

```bash
uv run python src/generate_combined_cached_viewer.py --models round --output-html data/combined_projection_functions_viewer.html
```

Сопоставление точек с ребрами, базовый режим:

```bash
uv run python src/match_full_circle_points_to_edges.py --model round --output-html output/round_full_circle_point_edge_matches.html --output-json output/round_full_circle_point_edge_matches.json
```

Сопоставление точек с ребрами при несовпадающих масштабах/уровнях `Z`:

```bash
uv run python src/match_full_circle_points_to_edges.py --model round --z-map adaptive --window-mode z-adaptive --output-html output/round_full_circle_point_edge_matches.html --output-json output/round_full_circle_point_edge_matches.json
```

Для быстрых проверок добавлять:

```bash
--max-contours 50 --no-progress
```

## Сопоставление точек с ребрами

Основной файл: `src/match_full_circle_points_to_edges.py`.

Текущая логика:

1. Из контуров строятся наблюдаемые `line_points` по сетке `z_levels`.
2. На каждом уровне выбираются точки для матчинга:
   - `--point-mode minima` - только локальные минимумы `fit_rms`, режим по умолчанию;
   - `--point-mode all` - все валидные точки.
3. Выбирается соответствующий уровень модели `z_reference`:
   - `--z-map identity` - тот же `Z`, режим по умолчанию;
   - `--z-map normalized` - линейное растяжение/сжатие диапазона `Z`;
   - `--z-map adaptive` - динамический монотонный выбор `z_reference` для каждого наблюдаемого уровня.
4. Модель сечется горизонтальной плоскостью `z = z_reference`; точки сечения хранят ссылки на `edge_id`.
5. Наблюдаемые точки послойно выравниваются к сечению модели через циклический сдвиг, возможный reverse и `affine`/`translation` transform.
6. Для каждой выровненной точки берутся ближайшие точки сечения и формируются кандидаты ребер.
7. Кандидаты оцениваются по расстоянию в XY и согласованности направления трека точки с направлением ребра.
8. Для каждого `point_index` Viterbi выбирает гладкую последовательность ребер по `Z`, штрафуя скачки на топологически далекие ребра.
9. В HTML/JSON пишутся итоговые `edge_id`, `edge_vertices`, `t_on_edge`, `distance_xy`, `direction_cost`, `confidence` и summary-метрики.

Важно: если модель топологически не похожа на объект, алгоритм все равно выдаст формальные `edge_id`, но результат может быть физически неверным. Если топология похожа, но размеры или уровни `Z` не совпадают, использовать `--z-map normalized` или `--z-map adaptive`.

## Визуализации

- Не плодить много HTML/JSON файлов для одной и той же проверки.
- Для временных экспериментов использовать понятные имена в `output/`, например `output/debug_match_round.html`.
- Если нужно сравнить режимы, лучше сохранять один JSON/HTML на режим с явным суффиксом: `_identity`, `_normalized`, `_adaptive`.
- Не коммитить тяжелые generated-файлы без явной просьбы пользователя.
- `plotly-2.35.2.min.js` может лежать рядом с HTML в `output/`; не дублировать его без причины.

## Практические проверки

Проверка импортов/синтаксиса без полного расчета:

```bash
uv run python -m compileall src
```

Проверка CLI основного матчера:

```bash
uv run python src/match_full_circle_points_to_edges.py --help
```

Короткий smoke-run матчера:

```bash
uv run python src/match_full_circle_points_to_edges.py --model round --max-contours 50 --no-progress --output-html output/smoke_point_edge_matches.html --output-json output/smoke_point_edge_matches.json
```

Короткий smoke-run full-circle viewer:

```bash
uv run python src/generate_full_circle_split_cached_viewer.py --models round --max-contours 50 --no-progress --output-html output/smoke_full_circle_split_viewer.html
```

## Работа с git

- Перед изменениями полезно смотреть `git status --short`.
- Не делать `git reset --hard`, `git checkout -- <file>` и другие destructive-команды без явной просьбы.
- Не включать в коммит чужие изменения.
- Если нужно изменить файл, который уже изменен пользователем, сначала прочитать актуальную версию и работать поверх нее.

## Кодстайл

- Предпочитать простые функции и dataclass-структуры, как в существующем коде.
- Не выносить один маленький эксперимент в несколько новых файлов.
- Для ручных правок использовать минимальный scoped diff.
- Комментарии добавлять только там, где они реально объясняют неочевидную геометрию, численную оптимизацию или формат данных.
- Сохранять численные пороги и допуски (`EPS`, `tol`) осознанно; при изменении объяснять, почему это безопасно.

## Эксперимент RMS-регионов и восстановления граней

Текущий экспериментальный файл: `src/reconstruct_faces_from_rms_regions.py`.

Цель эксперимента: восстановить плоскости/грани модели `round` из структуры `fit_rms` по уровням `Z`, затем построить модель как пересечение найденных полупространств и сравнить ее с `InitialModel`.

Идея пользователя:

1. На каждом уровне `z` есть функция `fit_rms` по циклическому индексу half-contour/window.
2. Высокие области `fit_rms` соответствуют переходу через грань, но максимум не всегда локальная точка, поэтому искать надо не локальные максимумы, а регионы выше порога.
3. Низкие области по обе стороны от peak-региона соответствуют точкам на ребрах этой грани, но минимум тоже часто область, а не монотонная точка, поэтому использовать надо регионы ниже порога.
4. Peak-регионы надо трекать по соседним уровням `z`; один устойчивый трек - кандидат на видимую область грани.
5. По low-регионам слева/справа от трека собираются точки, по ним фитится плоскость.
6. Итоговая модель должна строиться как пересечение полупространств этих плоскостей.

Что уже реализовано в `src/reconstruct_faces_from_rms_regions.py`:

- переиспользуется расчет `line_points`/`fit_rms` из `src/match_full_circle_points_to_edges.py`;
- сначала считаются несколько размеров окна из `--windows`, но дальше выбирается один адаптивный window на каждый уровень `z`, а не объединяются кандидаты всех окон;
- выделяются циклические peak-регионы выше абсолютного `--peak-threshold`;
- выделяются low-регионы ниже абсолютного `--low-threshold`;
- квантильные `--peak-quantile`/`--low-quantile` оставлены только как fallback, если absolute thresholds отключить отрицательными значениями;
- left/right low-регионы обязаны быть найдены с обеих сторон peak и быть не дальше `--max-low-distance`;
- peak-регионы трекаются по `z` по близости центра и ширины;
- из каждого left/right low-региона берется несколько representative-точек с минимальным RMS (`--low-region-samples`), а не вся широкая low-область;
- line_points, улетевшие далеко за bounding box shadow-контуров с запасом, не участвуют в фитинге;
- по representative-точкам left/right low-регионов фитятся плоскости кандидатов;
- hull кандидата рисуется строго в fitted-плоскости, а не исходными шумными точками;
- кандидаты сортируются по `candidate_score = plane_rms / sqrt(levels)`, затем по `hull_diameter`;
- ориентация полупространств задается через `--orientation-mode`:
  - `observed-points` - дефолтный режим, ориентирует нормали по большинству наблюдаемых `line_points`;
  - `model-vertices` - диагностический режим, ориентирует нормали по вершинам `InitialModel`;
  - `inside-point` - старый режим через одну среднюю внутреннюю точку;
- для `--orientation-mode model-vertices` есть диагностический support-фильтр `--model-support-max-outside-count`, который удаляет плоскости, если после ориентации они оставляют слишком много вершин `InitialModel` снаружи;
- reference-фильтр по `InitialModel`, merge похожих плоскостей и hull-фильтры сейчас убраны из основной логики;
- `InitialModel` в основном режиме используется в HTML только для сравнения, но в диагностическом `model-vertices` режиме может использоваться для ориентации нормалей и support-фильтра;
- модель строится как пересечение полупространств выбранных плоскостей: внутри считается сторона `normal · (x - point) <= tol`;
- после первого пересечения выполняется post-filter активных граней: если грань пересечения от кандидата сильно больше локального observed hull, кандидат удаляется и пересечение пересчитывается;
- HTML viewer показывает:
  - сетку 2x2: `Найденные плоскости + активные грани пересечения`, `Наложение`, `Исходная модель`, `Полученная модель`;
  - отдельные галочки `Найденные плоскости` и `Активные грани` для верхнего левого окна;
  - в `Наложение` нет найденных плоскостей-кандидатов, только исходная и восстановленная модели;
  - в `Наложение` исходная модель усилена opacity и черным wireframe;
  - все четыре окна должны синхронизировать камеру через `plotly_relayout`.

Основная команда для текущего эксперимента:

```bash
uv run python src/reconstruct_faces_from_rms_regions.py --model round --no-progress --output-html output/round_rms_face_regions.html --output-json output/round_rms_face_regions.json
```

Текущие важные дефолты:

```text
--windows 3,4,6
--peak-threshold 0.00055
--low-threshold 0.0002
--peak-quantile -1.0
--low-quantile -1.0
--max-low-distance 35.0
--low-region-samples 3
--max-plane-rms 0.08
--orientation-mode observed-points
--model-support-tol 0.03
--model-support-max-outside-count -1
--max-candidates 180
--max-active-face-hull-area-ratio 20.0
--max-active-face-extra-area 1.0
--active-face-filter-iterations 4
--intersection-inside-tol 0.03
--intersection-vertex-tol 0.02
```

Широкий набор окон `4,6,8,10,12,16,20,24,28` остается доступен как explicit CLI value для отдельных сравнений, но не является дефолтом validated reconstruction pipeline.

Каноническая packaging/freeze команда для воспроизводимого cross-model benchmark summary без нового пересчета reconstruction:

```bash
uv run python src/benchmark_reconstruction_quality.py \
  --input-json output/round_rms_w2_edge_tracks.json \
  --input-json /private/tmp/princess_angular_final_off.json \
  --input-json /private/tmp/radiant_angular_off_control.json \
  --input-json /private/tmp/pear_angular_balanced_off.json \
  --input-json /private/tmp/cushion_angular_off_control.json \
  --freeze-metadata-from output/rms_cross_model_angular_benchmark.json \
  --freeze-only \
  --output-json output/rms_cross_model_angular_benchmark.json \
  --output-html output/rms_cross_model_angular_benchmark.html
```

Обычный rerun benchmark evaluator не должен молча перезаписывать JSON, содержащий freeze metadata; для packaging/freeze использовать явный `--freeze-metadata-from`.

Последний результат для `round`:

```text
InitialModel: 508 vertices / 256 faces / 762 edges
peak_observations: 10852
tracks: 1328
raw_candidates_before_merge: 261
used planes before post-filter: 180
used planes after post-filter: 100
active_face_size_rejected: 80
intersection: 442 vertices / 100 faces
```

Последние выходные файлы:

- `output/round_rms_face_regions.html`;
- `output/round_rms_face_regions.json`.

Отдельный диагностический viewer для fixed `window=10`:

```bash
uv run python src/reconstruct_faces_from_rms_regions.py --model round --windows 10 --window 10 --no-progress --peak-threshold 0.00395 --low-threshold 0.00038 --orientation-mode model-vertices --model-support-max-outside-count 20 --max-active-face-hull-area-ratio -1 --max-active-face-extra-area -1 --output-html output/round_rms_face_regions_window10.html --output-json output/round_rms_face_regions_window10.json
```

Последний результат fixed `window=10`:

```text
peak_observations: 5477
tracks: 298
used planes after model support filter: 123
intersection: 417 vertices / 105 faces
orientation_mode: model-vertices
model_support_max_outside_count: 20
```

Важно: `window=10` плоскости визуально выглядят адекватно, но пересечение без диагностической ориентации/support-фильтра схлопывалось. С `--orientation-mode observed-points` было `41 vertices / 22 faces`, с `--orientation-mode model-vertices` без support-фильтра было `275 vertices / 99 faces`, с support-фильтром `20` стало `417 vertices / 105 faces`.

Что уже проверялось:

- `uv run python -m compileall src/reconstruct_faces_from_rms_regions.py`;
- HTML открывался через локальный `python3 -m http.server 8765 --bind 127.0.0.1 --directory output`;
- проверено, что четыре Plotly-панели рендерятся;
- проверено, что HTML содержит `plotPlanes`, `plotInitial`, `plotReconstructed`, `plotOverlay`;
- проверено, что overlay строится без `candidateTraces(limit)`;
- проверено, что HTML содержит `showPlanesInput` и `showActiveFacesInput`;
- на момент последней проверки статус viewer: `planes=100/100, intersection=442v/100f`.
- поле `Показать плоскостей` теперь по умолчанию равно числу выбранных кандидатов.

Важные выводы и проблемы:

- Adaptive window selection пока вырождается в window `4` на всех уровнях. Это лучше, чем объединять кандидаты всех окон, но критерий выбора окна еще слабый.
- Абсолютные пороги работают предсказуемее квантилей, но текущие значения подобраны под `round` и масштаб текущих `fit_rms`.
- Сбор всех точек из wide low-регионов был явной ошибкой: он создавал hull с диаметром `100+`. Сейчас берутся representative-точки с минимальным RMS, после чего максимальный `hull_diameter` среди выбранных кандидатов около `3.78`.
- Верхнее окно раньше показывало только маленькие observed hull-патчи, а пересечение строилось по бесконечным плоскостям. Поэтому аккуратные локальные плоскости могли давать странную большую модель. Сейчас сверху дополнительно рисуются активные clipped-грани пересечения.
- Добавлен post-filter `active_face_size`: он удалил 80 кандидатов, у которых активная грань пересечения была несоразмерно больше observed hull.
- Без reference-фильтра и merge похожих плоскостей результат стал более честным, но менее полным: сейчас `100` граней пересечения против `256` исходных.
- Простое добавление большего числа кандидатов после `180` снова ухудшает пересечение: при `220+` плоскостях часть несовместимых кандидатов схлопывает модель.
- Для fixed `window=10` основная проблема пересечения была в плоскостях, проходящих через объем: визуальный observed hull выглядел правильно, но бесконечное полупространство отрезало часть модели. Диагностический support-фильтр по `InitialModel` это подтверждает.
- Следующее улучшение должно повышать совместимость набора плоскостей без обращения к `InitialModel`.

Как улучшать дальше:

1. Улучшить критерий adaptive window: сейчас выбран window `4` на всех уровнях, потому что малые окна дают минимальный RMS. Нужно выбирать не минимальный RMS, а самый информативный устойчивый масштаб.
2. Разделить кандидаты рундиста, короны и павильона. Сейчас плоскости разных зон попадают в общий пул и конкурируют в одном пересечении.
3. Улучшить трекинг peak-регионов: учитывать не только центр/ширину, но и overlap интервалов, соседние low-регионы, амплитуду peak и гладкость по `z`.
4. Проверять, не склеивает ли трекинг разные физические peak-регионы при длинных треках.
5. Ввести scoring совместимого набора плоскостей перед пересечением:
   - плоскость должна быть активной гранью пересечения;
   - не должна отрезать большую наблюдаемую часть;
   - должна иметь достаточную поддержку по уровням `z`;
   - должна иметь малый plane RMS;
   - должна не создавать чрезмерно большие грани, проходящие поперек модели.
6. Сделать итеративное пересечение: добавлять плоскости по одной и отклонять те, которые резко уменьшают объем/число активных граней или дают неадекватные сечения.
7. Заменить диагностический support-фильтр по `InitialModel` внутренним критерием: плоскость не должна резко противоречить наблюдаемым сечениям/контурам и не должна проходить через объем.
8. Для диагностики добавить в HTML фильтры по конкретным окнам `window`, зонам `z`, `plane_rms`, `candidate_score`, `hull_diameter`, `low_distance_max`, `support_outside_count`.
9. Для финального сравнения добавить метрики между восстановленной и исходной моделью:
   - распределение расстояний вершин восстановленной модели до поверхности `InitialModel`;
   - распределение расстояний вершин `InitialModel` до восстановленной поверхности;
   - объем/габариты;
   - число активных плоскостей.

Полезные команды для следующего чата:

```bash
uv run python src/reconstruct_faces_from_rms_regions.py --help
uv run python src/reconstruct_faces_from_rms_regions.py --model round --no-progress --output-html output/round_rms_face_regions.html --output-json output/round_rms_face_regions.json
uv run python -m compileall src/reconstruct_faces_from_rms_regions.py
```

Быстрый, но менее показательный smoke-run:

```bash
uv run python src/reconstruct_faces_from_rms_regions.py --model round --max-contours 50 --no-progress --output-html output/smoke_rms_face_regions.html --output-json output/smoke_rms_face_regions.json
```

Для проверки HTML локально:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory output
```
