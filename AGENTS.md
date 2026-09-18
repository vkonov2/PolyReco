# AGENTS.md

Инструкции для работы в `PolyReco`. Справочные рецепты подключаются только по задаче.

## Границы работы

- Всегда отвечать на русском языке.
- Перед изменениями проверить `git status --short`, прочитать затронутый код и сохранить его стиль.
- Не откатывать и не перезаписывать чужие изменения; работать только в согласованном объёме.
- Не делать commit/push и не добавлять тяжёлые generated-файлы без явной просьбы.
- Не расширять оформление, отчёт, диагностику или упаковку до изменения production-алгоритма без задания.
- Обычные локальные шаги согласованной задачи выполнять без повторного запроса разрешения.
- Уточнить, если нужны не предусмотренное задачей внешнее изменение состояния, расширение задачи или риск потери данных.
- Использовать `rg` / `rg --files`; не добавлять крупные зависимости без необходимости.
- Предпочитать небольшие scoped-правки, простые функции и существующие dataclass-структуры.
- Численные допуски и пороги менять осознанно, объясняя влияние на геометрию.

## Данные и окружение

- Python `>=3.13`, `uv`; зависимости и entry points — в `pyproject.toml`.
- Запуск: `uv run python src/<script>.py ...`; `uv sync` нужен только при настройке/обновлении окружения.
- Модели: `round`, `princess`, `radiant`, `pear`, `cushion` в `data/<model>/`.
- `InitialModel` — эталон; `shadow/merged-cont*` — наблюдаемые контуры; OBJ — дополнительные модели.

## Карта кода

- `src/polyreco/io.py`, `model_data.py`, `contour.py` — входные данные и структуры.
- `src/reconstruct_faces_from_rms_regions.py` — CLI и orchestration RMS-пайплайна; часть стадий остаётся здесь.
- `src/polyreco/rms_candidates.py` — регионы, треки, fit плоскостей и face candidates.
- `src/polyreco/rms_w2.py` — W2-сегменты, кластеризация и геометрия пар.
- `src/polyreco/rms_selection.py` — support/ranking, совместимость, LP activity и outside.
- `src/polyreco/rms_mesh.py` — полупространства, edge-clip, топология и объём.
- `src/polyreco/rms_reprojection.py` — проекции, расстояния и полярное сравнение контуров.
- `src/polyreco/rms_output.py` — reconstruction JSON и HTML.
- `src/polyreco/rms_oracle_loss_diagnostics.py` — oracle-only анализ и маршрутизация diagnostic scopes.
- `src/match_full_circle_points_to_edges.py` — observed line points и сопоставление с рёбрами эталона.
- `src/generate_full_circle_split_cached_viewer.py` — full-circle viewer и его кэши.
- `src/benchmark_reconstruction_quality.py` — posthoc-метрики и freeze упаковка.
- `src/generate_all_models_reconstruction_viewer.py` — общий viewer из benchmark payload / model JSON.

## Что читать по задаче

- Для локальной правки читать затронутый код; всю карту проекта и все diagnostics заранее не загружать.
- Для матчинга и старых viewer-рецептов — соответствующий раздел [справочника](docs/polyreco_workflows.md).
- Для RMS-истории — [архив эксперимента](docs/polyreco_workflows.md#архив-rms-эксперимента); это история, не текущий roadmap.
- Для восстановления по известному шаблону — [дорожная карта](docs/template_reconstruction_roadmap.md): постановка, этапы и критерии проверки; выполнять этапы в объёме текущей задачи.
- Для воспроизведения/freeze сначала читать выбранный artifact и его `parameters`, затем нужный код упаковки.
- Старые результаты, планы и команды из справочника не являются поручением запускать эксперименты.

## Артефакты и воспроизводимость

- Не плодить HTML/JSON: переиспользовать выбранный путь; по умолчанию generated-файлы держать в `output/`.
- Для smoke/эксперимента задавать отдельные output **и cache** пути; не заменять ими полноценные данные.
- Golden `output/round_rms_w2_edge_tracks.json/html` и selected cross-model artifacts не перезаписывать
  при диагностике, smoke или оформлении без прямого задания обновить именно этот результат.
- `output/` может игнорироваться git: отсутствие файлов в `git status` не означает, что они не изменены.
- Baseline определяется полным `parameters` выбранного JSON, данными и семантикой метрик, а не CLI defaults.
- Текущий default `--windows 3,4,6` не заменяет полный набор параметров golden; широкий набор — explicit option.
- В production reconstruction `InitialModel` не должен участвовать в generation/selection/gates.
  Reference-driven режимы допустимы только как явно обозначенная диагностика, не как non-oracle результат.
- Не смешивать candidate-level ceiling с final active mesh; topology, canonical matching, finite footprint,
  trusted outside/lost-Z и oracle/no-oracle parity — отдельные проверки.
- Сравнивать outside только при одинаковом point set, tolerances и JSON path; viewer cloud не равен trusted cloud.
- Freeze с существующими метаданными — только явный `--freeze-metadata-from ... --freeze-only`
  и существующие проверенные `--input-json`. Не восстанавливать пропавшие temp-файлы пересчётом без задания.
- Общий viewer умеет читать embedded payload benchmark JSON; `/private/tmp` не должен быть runtime-зависимостью.

## Соразмерные проверки

- Выбирать проверки по изменению, а не запускать все команды справочника как обязательный checklist.
- Только текст/инструкции: проверить diff, ссылки и смысл; reconstruction, smoke и compileall не нужны.
- Python/CLI: проверить синтаксис затронутых файлов; `--help` нужен при изменении CLI.
- HTML/визуализация: проверить затронутый экран, подписи и взаимодействия без пересчёта геометрии, если возможно.
- Алгоритм/геометрия: начать с ограниченного отдельного прогона; полный контроль нужен для утверждения о
  сохранении/улучшении качества, но не для косметической правки.
- Smoke подтверждает работоспособность, не качество восстановления. Не сравнивать его с полным golden.
- Не повторять успешную проверку без относящихся к ней изменений. При изменении данных/параметров проверить заново.
- Завершать после согласованного результата и достаточной проверки; непроверенное явно указать.
  Не продолжать исторический roadmap и не полировать бесконечно без нового задания.

## Изолированный smoke full-circle viewer

Запускать только если он нужен для затронутой функциональности. Команда из корня проекта:

```bash
uv run python src/generate_full_circle_split_cached_viewer.py \
  --models round --max-contours 50 --no-progress \
  --cache-json output/smoke_full_circle_split_cache.json \
  --output-html output/smoke_full_circle_split_viewer.html
```

Если добавляется precompute окон, также задать отдельный `--windows-cache-json`;
не использовать основной кэш для сокращённого smoke.
