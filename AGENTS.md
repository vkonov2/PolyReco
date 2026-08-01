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

