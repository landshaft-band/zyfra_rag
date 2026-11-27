# Локальный RAG-помощник «Цифра. Диспетчер»

Спецификация текущего прототипа и инструкция для продолжения работы в новом диалоге. Проект отвечает на вопросы техподдержки по Markdown-документации и решённым тикетам. Индексация и поиск работают локально. Для генерации ответа используется настраиваемая модель через OpenAI-совместимый HTTP API.

## Что уже работает

- Разбор двух Markdown-источников, экспорт в JSONL и собственный чанкинг с учётом токенов.
- Локальные эмбеддинги multilingual E5, индексация в Qdrant, плотный и лексический поиск, гибридная выдача через RRF.
- Генерация ответа с цитатами, проверка структуры ответа и резервная выдача найденных фрагментов.
- CLI и локальный FastAPI с русскоязычным веб-интерфейсом и Swagger. `POST /api/ask` и CLI могут обрабатывать отдельные вопросы.
- При настроенном PostgreSQL веб-чат сохраняет реплики и цитаты, загружает историю и учитывает предыдущие сообщения при поиске.
- Веб-чат отправляет вопросы как фоновые задания. PostgreSQL хранит их состояние, Kafka доставляет события готовности, а Redis Streams раздаёт задания RAG-воркерам.
- Профиль Compose `app` запускает API, воркер, публикатор и диспетчер из одного Docker-образа.
- Манифесты Kubernetes разворачивают тот же образ и отдельные хранилища в локальном кластере kind.
- Офлайн-оценка поиска, опциональное LLM-реранжирование в вебе и CLI, оценка ответов через Ragas и трассировка через Langfuse.
- Guardrails AI проверяет вход и ответ. В Docker Compose по умолчанию включено блокирование.

## Данные и индексация

Исходники на текущей машине лежат в `C:/Users/winda/Downloads`. Их имена — `support_tickets_digit_dispatcher_200 (1).md` и `zyfra_product_support_technical_reference (2).md`. Ниже пути относительно корня репозитория.

| Файл | Содержимое |
| --- | --- |
| [ticket_sources.jsonl](data/processed/ticket_sources.jsonl) | 200 записей с решёнными тикетами |
| [document_sources.jsonl](data/processed/document_sources.jsonl) | 130 разделов документации |
| `corpus_chunks.jsonl` | Генерируется при индексации. Текущий корпус даёт 553 чанка, из них 200 из тикетов и 353 из документации |
| `data/processed/bm25/` | Генерируемый лексический словарь и статистика BM25 для конкретной версии гибридной коллекции |
| [data/eval](data/eval/) | Наборы контрольных вопросов на 14 и 100 кейсов. Отчёты прогонов сохраняются локально |

`data/processed/sources.jsonl` и `data/processed/chunks.jsonl` относятся к раннему тестовому набору. Полный корпус строится из двух экспортов из таблицы.

1. [Парсер тикетов](src/rag/ingest/tickets.py) выделяет карточки по заголовкам с номером тикета. [Парсер документации](src/rag/ingest/documents.py) извлекает разделы с путём заголовков и данными для ссылки на источник.
2. [Пайплайн индексации](src/rag/ingest/pipeline.py) читает JSONL. [Токенный чанкинг](src/rag/ingest/token_chunking.py) сохраняет границы источников и подзаголовков, по возможности режет по абзацам и использует перекрытие для длинного текста.
3. [E5](src/rag/retrieval/e5.py) считает векторы локально на CPU. Для текста применяется префикс `passage:`, для вопроса — `query:`. Ревизия модели закреплена в [tokenization.py](src/rag/ingest/tokenization.py).
4. `rag build-index` сохраняет чанки и плотные векторы и публикует алиас `chunks_corpus`. `rag build-hybrid` использует сохранённые векторы, добавляет BM25 и публикует `chunks_hybrid`. Алиасы указывают на версионированные коллекции. Гибридному поиску нужна соответствующая коллекции статистика на диске. [Плотный индекс](src/rag/retrieval/indexing.py) · [Гибридный индекс](src/rag/retrieval/hybrid_index.py)

## Как обрабатывается вопрос

1. Веб с PostgreSQL отправляет `POST /api/conversations/{id}/jobs` с вопросом и ключом запроса. Воркер получает задание, загружает историю и при необходимости переформулирует вопрос перед поиском. Веб без PostgreSQL использует `POST /api/ask`. Оба режима и команда `rag ask` используют общий [AskService](src/rag/generation/service.py).
2. [HybridRetriever](src/rag/retrieval/hybrid.py) запускает поиск по E5 и BM25 в Qdrant, берёт до 50 результатов из каждой ветки и объединяет их по позициям через Reciprocal Rank Fusion.
3. При включённом реранжировании [AskService](src/rag/generation/service.py) берёт до 100 результатов RRF, убирает точные повторы и почти одинаковые фрагменты одного раздела, затем передаёт до 20 уникальных кандидатов в [LLM-реранкер](src/rag/evaluation/rerank.py). По умолчанию реранкер выбирает до восьми чанков. Реранкер может вернуть меньше восьми фрагментов, включая пустой список. Его выдача снова проверяется на дубли. При пустом отборе помощник запрашивает уточнение без вызова модели генерации. При ошибке и в выключенном режиме сервис использует первые восемь уникальных чанков RRF. В Langfuse у этапов hybrid-search и rerank видны счётчики дублей. [Сборщик контекста](src/rag/generation/context.py) присваивает переданным чанкам идентификаторы `S1`…`S8` в порядке выдачи. [Клиент модели](src/rag/generation/client.py) отправляет вопрос и контекст в `/chat/completions`.
4. [Валидатор](src/rag/generation/answer.py) разбирает JSON модели и проверяет ссылки на источники. Модель может вернуть `answer`, `clarify` или `no_answer`. При проблеме с генерацией или проверкой сервис возвращает `search_only` с найденными фрагментами.
5. [FastAPI](src/rag/web/app.py) сериализует результат, а [веб-страница](src/rag/web/static/index.html) показывает ответ и карточки источников. Модель возвращает массив `answer_items`, а HTTP-ответ содержит массив `items`. Для запроса с диалогом реплики и цитаты сохраняются в PostgreSQL.

## Где модель и промпты

| Назначение | Где менять |
| --- | --- |
| Генерация ответа помощника | `SYSTEM_PROMPT` в [generation/client.py](src/rag/generation/client.py). Здесь заданы поведение помощника и контракт JSON с цитатами. |
| Формат передаваемых источников | [generation/context.py](src/rag/generation/context.py) и сборка запроса в [generation/client.py](src/rag/generation/client.py). |
| Переформулировка вопроса по истории | `REWRITE_PROMPT` в [generation/history.py](src/rag/generation/history.py). |
| LLM-реранжирование | `PROMPT` в [evaluation/rerank.py](src/rag/evaluation/rerank.py). Его используют помощник и оценка. |
| LLM-судья | Системная инструкция адаптера в [evaluation/ragas_metrics.py](src/rag/evaluation/ragas_metrics.py). Сами метрики используют шаблоны установленной версии Ragas. |

Промпты сейчас хранятся в Python. Чтобы сохранить новую версию, отредактируй константу, сохрани файл и зафиксируй изменение в Git. Если меняешь форму ответа или имена полей, проверь валидатор и контракт FastAPI. Сборка Python после правки не требуется. Если веб уже запущен, останови его через `Ctrl+C` в его терминале и запусти снова.

`ChatClient.from_environment()` читает конфигурацию из окружения процесса. При запуске через `uv run --env-file` используется указанный файл, например `C:/Users/winda/AppData/Local/rag/polza.env`. Имя файла произвольное. Содержимое для нового компьютера или нового провайдера заполняется локально.

```dotenv
RAG_LLM_BASE_URL=https://api.example.com/v1
RAG_LLM_MODEL=provider/model-id
RAG_LLM_API_KEY=YOUR_API_KEY
```

Укажи базовый адрес по инструкции провайдера. Клиент сам добавляет `/chat/completions`. Модель указывается точным идентификатором из API провайдера. Настройки находятся на стороне Python и ключ хранится вне репозитория. После правки env перезапусти веб той же командой. Для изменения провайдера, модели генерации или промпта существующий индекс можно использовать повторно.

### Настройки реранжирования

Добавь в тот же env-файл:

```dotenv
RAG_RERANK_ENABLED=true
RAG_RERANK_TOP_N=20
RAG_RERANK_TOP_K=8
# Необязательная конфигурация отдельной модели
# RAG_RERANK_BASE_URL=https://ranking.example.com/v1
# RAG_RERANK_MODEL=provider/ranking-model
# RAG_RERANK_API_KEY=YOUR_RERANK_API_KEY
```

Если `RAG_RERANK_ENABLED` отсутствует, веб и `rag ask` обходятся без реранжирования. Без отдельных параметров реранкер использует адрес, модель и ключ генерации. При смене адреса ключ укажи отдельно, если он нужен провайдеру. `TOP_N` задаёт число кандидатов в пределах 1–100, `TOP_K` задаёт число отбираемых чанков в пределах 1–8 и не может быть больше `TOP_N`. При ошибке реранжирования помощник использует до восьми первых результатов RRF независимо от `TOP_K`. В логах доступны модель, количество и ID входных и выходных чанков, длительность и причина fallback.

### Langfuse

Создай проект в Langfuse и добавь его настройки в локальный env-файл. Для Langfuse Cloud подставь адрес своего региона, для собственного инстанса — его базовый URL.

```dotenv
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=YOUR_LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY=YOUR_LANGFUSE_SECRET_KEY
LANGFUSE_TRACING_ENABLED=true
```

Останови веб через `Ctrl+C`, установи зависимости и снова запусти сервис. Если в этом же окружении будешь запускать Ragas, добавь `--extra evaluation` к команде `sync`.

```powershell
uv.exe sync --locked --extra observability
uv.exe run --no-sync --env-file 'C:/Users/winda/AppData/Local/rag/polza.env' rag serve
```

После вопроса открой **Tracing** в проекте Langfuse и найди `rag.ask` с этапами `hybrid-search`, `rerank` при включённой настройке и `answer-generation` при наличии контекста. Запуск `rag eval` создаёт наблюдение `rag.eval.case` на каждый кейс с позициями источников и метриками Ragas, когда они рассчитаны. По умолчанию трассировка передаёт служебные данные, включая ID чанков и доступные счётчики токенов. Без ключей Langfuse или при `LANGFUSE_TRACING_ENABLED=false` экспорт отключён.

### Подробные трассы и стоимость

Счётчики токенов отправляются из ответа провайдера. В метаданных каждого вызова модели `cost_rub` означает фактическую сумму в рублях, если провайдер её вернул. Явная сумма `cost_usd` также заполняет поле стоимости Langfuse в USD. Для провайдеров без суммы можно задать тарифы на миллион токенов, привязанные к точному имени модели. Расчёт отображается отдельно в `estimated_cost_rub` или `estimated_cost_usd`. Пример со схематическими тарифами.

```powershell
$env:RAG_MODEL_PRICES_JSON = '{"provider/model":{"currency":"rub","input_per_million":50,"output_per_million":100}}'
```

По умолчанию тексты не экспортируются. Для отладки на тестовом корпусе включи их для API и воркера в Compose.

```powershell
$env:RAG_TRACE_CONTENT = 'true'
$env:RAG_MODEL_ENV_FILE = 'C:/Users/winda/AppData/Local/rag/polza.env'
docker compose --profile app up -d --no-build --wait
```

В `guardrails.input` виден проверенный вопрос, в `hybrid-search` — поисковый запрос, позиции и тексты чанков. В `question-rewrite`, `rerank`, `answer-generation` и `ragas-judge` видны сообщения, отправленные провайдеру. Сырой ответ чат-бота и разобранные пункты появляются в `answer-validation` после выходной проверки. При блокировке сырой ответ исключается. Из полей с текстом удаляются известные ключи и строки с типичными признаками учётных данных. Для выключения опции задай `RAG_TRACE_CONTENT=false` и снова запусти Compose.

### Guardrails AI

[AskService](src/rag/generation/service.py) проверяет исходный вопрос перед переформулировкой и поиском, а текст ответа и уточнений — перед возвратом в API или воркер. [Валидатор Guardrails AI](src/rag/safety_validators.py) использует локальные правила для прямых команд обойти инструкции и запросов на раскрытие системного промпта. На выходе он ищет совпадение нескольких характерных строк системного промпта. Проверки выполняются без отдельного модельного вызова.

В Compose по умолчанию включён `RAG_GUARDRAILS_MODE=enforce`. Срабатывание на входе останавливает запрос до поиска и вызовов модели. Срабатывание на выходе заменяет ответ сообщением о блокировке до показа и сохранения. Заблокированный обмен виден в чате со статусом `blocked` и исключается из контекста следующих запросов. При ошибке валидатора пользователь видит сообщение о временной недоступности проверки. Для смены режима задай `$env:RAG_GUARDRAILS_MODE` перед запуском Compose. `observe` только регистрирует срабатывания, `off` отключает проверки. При запуске Python напрямую без этой переменной используется `observe`.

Для асинхронного веб-чата в Langfuse ищи `guardrails.input` внутри `rag.submit`. После успешной проверки появится трасса `rag.job → rag.ask → guardrails.output`. Трассы приёма и выполнения связаны через `request_id`. Метаданные `input_guard_at=submit` в `rag.ask` показывают, что проверка состоялась при приёме задания. Входная блокировка возвращает HTTP 403 до создания задания и события outbox. При недоступной проверке в режиме `enforce` API возвращает HTTP 503. Старые задания без отметки о проверке воркер проверяет перед `rag.ask`.

В метаданных проверок есть решение `pass`, `flag` или `error`, причина и время. Чтобы отправлять трассы приёма из Compose в Langfuse, сохрани настройки `LANGFUSE_*` в локальном `.env.api` или укажи файл через `RAG_API_TRACE_ENV_FILE`. Срабатывания видны в логах `docker compose --profile app logs api worker`. Журнал Compose содержит только метаданные проверок.

Текущие правила покрывают прямые формулировки. На небольшой дополнительной выборке из 20 синтетических примеров отмечены восемь из десяти попыток и один обычный запрос с цитатой из журнала. Два косвенных запроса пропущены. Проверка чанков пока не включена.

## Самостоятельный запуск на Windows

### Стенд целиком в Docker Compose

Запусти Docker Desktop. Из корня проекта укажи файл конфигурации модели и подними профиль `app`. Локальный Python для этого варианта не нужен.

```powershell
$env:RAG_MODEL_ENV_FILE = 'C:/Users/winda/AppData/Local/rag/polza.env'
docker compose --profile app up -d --build
docker compose --profile app ps -a
```

- [Чат в контейнере](http://127.0.0.1:8002/) и [Swagger](http://127.0.0.1:8002/docs). При необходимости переопредели порт через `RAG_WEB_PORT`.
- Образ из [Dockerfile](Dockerfile) собирается по `uv.lock` и запускается от отдельного пользователя. Воркер получает файл настройки модели, а API читает собственный файл настроек Langfuse. Без файла модели можно проверить поиск и сохранение результата `search_only`.
- `migrate` применяет миграции, `queue-init` готовит топик Kafka и группу Redis, `index-init` проверяет гибридный индекс. Успешное состояние этих разовых контейнеров — `Exited (0)`. Долгоживущие процессы — `api`, `worker`, `publisher`, `dispatcher`.
- Если Qdrant и подходящий BM25-файл уже есть, `index-init` использует их. С пустым Qdrant он собирает индекс из двух экспортов в `data/processed`. Экспорты монтируются только для чтения. Сгенерированные чанки и BM25 лежат в томе `index_artifacts`, кэш E5 — в `e5_cache`. При первом построении или запросе может потребоваться скачать E5. Изменения исходных экспортов применяются явной переиндексацией.
- Профиль использует текущие тома PostgreSQL и Qdrant проекта `rag-local`. У контейнерного API и фоновых процессов внутренние адреса сервисов Compose. При переходе на контейнерный режим завершай ранее запущенные вручную публикатор, диспетчер и воркер, чтобы на одном стенде было проще отследить, какой процесс взял задание.

Проверка и управление процессами

```powershell
docker compose --profile app logs --tail 50 migrate queue-init index-init
docker compose --profile app logs -f api worker publisher dispatcher
# После правки файла модели пересоздать только воркер
docker compose --profile app up -d --no-deps --force-recreate worker
# Остановить приложение, сохранив инфраструктуру и данные
docker compose --profile app stop api worker publisher dispatcher
```

Для отдельного стенда с чистыми томами укажи другое имя проекта через `docker compose -p` и переопредели `RAG_WEB_PORT`, `RAG_POSTGRES_PORT`, `RAG_QDRANT_PORT`, `RAG_REDIS_PORT` и `RAG_KAFKA_PORT`. `RAG_CORPUS_DIR` позволяет задать другой каталог экспортов.

Чтобы применить обновлённые экспорты в Docker, собери новую версию индекса из каталога `/app/corpus`, затем перезапусти процессы чтения.

```powershell
docker compose --profile app run --rm index-init --qdrant-url http://qdrant:6333 build-index --sources /app/corpus/ticket_sources.jsonl /app/corpus/document_sources.jsonl
docker compose --profile app run --rm index-init --qdrant-url http://qdrant:6333 build-hybrid
docker compose --profile app restart api worker
```

### Локальный Kubernetes в kind

Для этого сценария нужны Docker Desktop с Linux-контейнерами, `kubectl` и `kind`. Команды используют отдельный кластер `rag` с контекстом `kind-rag`. Если он уже существует, начни с загрузки образов. На текущей машине `kind.exe` доступен в каталоге `C:/Users/winda/AppData/Local/rag/tools`. [Установка kind](https://kind.sigs.k8s.io/docs/user/quick-start/)

```powershell
$ragKind = Join-Path $env:LOCALAPPDATA 'rag/tools/kind.exe'
& $ragKind create cluster --name rag --wait 150s
docker build --provenance=false --sbom=false -t rag-local-app:k8s-dev .
& $ragKind load docker-image --name rag rag-local-app:k8s-dev
./k8s/deploy.ps1 -ModelEnvFile 'C:/Users/winda/AppData/Local/rag/polza.env'
```

После обновления Python-кода пересобери образ с тем же тегом, снова загрузи его в kind и повтори `deploy.ps1`. Скрипт пересоздаёт разовые задания и запускает обновление Pod. Без `-ModelEnvFile` на чистом стенде работает локальный поиск с результатом `search_only`. Файл модели с ключами используется для создания Kubernetes Secret и доступен только воркеру приложения.

Для поэтапного запуска `deploy.ps1 -Stage Infra` подготовит хранилища, конфигурацию и инфраструктурные сервисы, а `deploy.ps1 -Stage App` выполнит подготовительные задания и запустит приложение. Флаги сборки в примере убирают метаданные, с которыми импорт в локальный kind на этой машине завершался ошибкой. [Известная проблема импорта kind](https://kind.sigs.k8s.io/docs/user/known-issues/#unable-to-kind-load-docker-images)

```powershell
kubectl --context kind-rag -n rag get deployment,statefulset,job,pvc
kubectl --context kind-rag -n rag logs job/index-init
kubectl --context kind-rag -n rag logs -f deployment/worker
kubectl --context kind-rag -n rag port-forward svc/api 8003:8000
```

Открой [чат Kubernetes](http://127.0.0.1:8003/) или [Swagger](http://127.0.0.1:8003/docs). Команда `port-forward` должна оставаться запущенной. Если Pod API сменился, запусти её снова. Порт Compose остаётся 8002.

- [k8s/infra](k8s/infra/) содержит четыре одноузловых `StatefulSet` и постоянные тома для данных. История Kubernetes хранится в отдельном PostgreSQL. Кластер заново индексирует загруженный корпус.
- [k8s/bootstrap](k8s/bootstrap/) запускает `migrate`, `queue-init` и `index-init` как отдельные `Job`. Скрипт ждёт их завершения перед применением приложения.
- [k8s/app](k8s/app/) содержит `Deployment` API, воркера, публикатора и диспетчера. Внешний доступ на учебном стенде идёт через локальный `port-forward`. API сообщает готовность по `/api/health`.
- Два текущих экспорта помещаются в `rag-corpus` ConfigMap. Если их суммарный размер превысит 1 МиБ, для корпуса нужно отдельное хранилище. BM25 и кэш E5 используют PVC. На чистом стенде без кэша первое построение индекса скачивает E5.

Проверить восстановление Pod можно командой `kubectl --context kind-rag -n rag rollout restart deployment/worker`. Статусы задач доступны через тот же API. Этот профиль рассчитан на один локальный узел. При разнесении Pod по нескольким узлам потребуется заменить локальные тома BM25 и модели хранилищем, доступным этим Pod.

### Запуск Python-процессов на хосте

Нужны PowerShell, `uv` и запущенный Docker. Проект закреплён на Python 3.13 через `.python-version` и `pyproject.toml`. `uv sync` подберёт интерпретатор и при необходимости пересоздаст окружение проекта. Все команды запускай из корня проекта. В другом расположении подставь свой путь.

```powershell
Set-Location 'C:/Users/winda/Documents/rag'
docker compose up -d qdrant
uv.exe sync --locked
```

Команда `uv sync` нужна при первом запуске и при изменении зависимостей. Если старый веб-процесс держит `rag.exe` открытым, сначала останови его. Docker Compose поднимает Qdrant на `127.0.0.1:6333` и хранит коллекции в Docker volume. Python и веб работают в локальном окружении `.venv`.

Для нового или пустого Qdrant выполни обе команды по порядку. При готовых коллекциях и соответствующем файле BM25 можно сразу запускать веб.

```powershell
uv.exe run --no-sync rag build-index
uv.exe run --no-sync rag build-hybrid
```

При первом построении E5 скачивается в локальный кэш. Обычный запуск веба уже готового индекса выглядит так.

```powershell
uv.exe run --no-sync --env-file 'C:/Users/winda/AppData/Local/rag/polza.env' rag serve
```

- [Веб-чат](http://127.0.0.1:8000/)
- [Swagger UI](http://127.0.0.1:8000/docs)
- [OpenAPI JSON](http://127.0.0.1:8000/openapi.json)
- [Готовность поискового индекса](http://127.0.0.1:8000/api/health). Эта ручка проверяет доступность коллекции, а проверка генерации выполняется отдельным вопросом.

Поиск без обращения к внешней модели можно проверить во втором терминале. Проверка через `rag ask` использует настроенный API и может расходовать баланс.

```powershell
uv.exe run --no-sync rag search 'Как открыть архив?' --mode hybrid --limit 3
uv.exe run --no-sync --env-file 'C:/Users/winda/AppData/Local/rag/polza.env' rag ask 'Как открыть архив?'
```

Для проверки резервного режима можно отдельно запустить веб без env-файла в терминале, где переменные `RAG_LLM_*` отсутствуют. Если порт занят, передай `rag serve --port 8001`.

### Хранилище диалогов в PostgreSQL

Локальный Compose запускает PostgreSQL 17 на `127.0.0.1:5433` с постоянным томом `postgres_data`. Код доступа к данным находится в [storage.py](src/rag/conversations/storage.py), начальная схема — в [001_conversations.sql](src/rag/conversations/migrations/001_conversations.sql), маршруты — в [web/conversations.py](src/rag/web/conversations.py).

Запусти PostgreSQL и Qdrant, установи зависимости и выполни миграцию. Ниже DSN с локальными параметрами из Compose. Если переопределял `RAG_POSTGRES_PASSWORD`, укажи соответствующий пароль в `RAG_POSTGRES_DSN`. При установленных Langfuse или Ragas добавь к `uv sync` используемые флаги `--extra observability` и `--extra evaluation`.

```powershell
docker compose up -d --wait postgres qdrant
uv.exe sync --locked
$env:RAG_POSTGRES_DSN = 'postgresql://rag:rag-local-dev@127.0.0.1:5433/rag'
uv.exe run --no-sync rag migrate
uv.exe run --no-sync --env-file 'C:/Users/winda/AppData/Local/rag/polza.env' rag serve --port 8001
```

Команду `rag migrate` можно запустить повторно. Веб по-прежнему использует существующий гибридный индекс Qdrant. Если индекса ещё нет, сначала выполни команды из раздела запуска.

Проверь [Swagger](http://127.0.0.1:8001/docs) или выполни во втором терминале:

```powershell
$ragApi = 'http://127.0.0.1:8001/api/conversations'
$chat = Invoke-RestMethod -Uri $ragApi -Method Post -SessionVariable ragBrowser
Invoke-RestMethod -Uri $ragApi -WebSession $ragBrowser
Invoke-RestMethod -Uri "$ragApi/$($chat.id)" -WebSession $ragBrowser
```

`POST /api/conversations` создаёт пустой диалог. `GET /api/conversations` перечисляет диалоги текущего браузера, `GET /api/conversations/{id}` возвращает их сообщения в порядке записи. Для локальной изоляции браузер получает HTTP-only cookie, PostgreSQL хранит её хеш. UUID чужого диалога вернёт `404` без соответствующей cookie.

Веб-чат создаёт диалог при первом сообщении. После перезагрузки страницы история восстанавливается. Через список диалогов можно открыть прежний чат, а кнопка **Новый чат** начинает отдельную историю. Без `RAG_POSTGRES_DSN` веб использует режим отдельных вопросов. Синхронный `POST /api/conversations/{id}/turns` доступен для ручных проверок, пока в диалоге нет активных заданий.

Для уточнения [history.py](src/rag/generation/history.py) передаёт текущий вопрос и до шести предыдущих сообщений модели генерации. История ограничена 6000 символами. Модель формулирует самостоятельный вопрос для поиска, реранжирования и ответа. При сбое модельного API или невалидном JSON используется исходный текст. В Langfuse переформулировка отображается отдельным этапом `question-rewrite` со служебными метаданными.

Чтобы запустить тесты на реальной базе, задай `RAG_TEST_POSTGRES_DSN` для локального стенда и выполни `uv.exe run --no-sync python -B -m unittest discover -s tests -v`. Интеграционные тесты создают отдельную временную схему в указанной базе.

### Очередь Redis Streams и диспетчеризация Kafka

На локальном стенде PostgreSQL доступен на `127.0.0.1:5433`, Kafka на `127.0.0.1:9092`, Redis на `127.0.0.1:6379`. Python-процессы запускаются из корня проекта. Если работаешь с Langfuse или Ragas, укажи используемые флаги `--extra` при установке зависимостей.

```powershell
docker compose up -d --wait postgres qdrant kafka redis
uv.exe sync --locked --extra observability
$env:RAG_POSTGRES_DSN = 'postgresql://rag:rag-local-dev@127.0.0.1:5433/rag'
uv.exe run --no-sync rag migrate
uv.exe run --no-sync rag queue-init
```

Укажи тот же `RAG_POSTGRES_DSN` в каждом из четырёх терминалов. Следующие команды запускаются по одной на терминал. Только воркеру нужна конфигурация моделей для обработки чата.

```powershell
uv.exe run --no-sync rag queue-publisher
uv.exe run --no-sync rag queue-dispatcher
uv.exe run --no-sync --env-file 'C:/Users/winda/AppData/Local/rag/polza.env' rag queue-worker
uv.exe run --no-sync rag serve --port 8001
```

Адреса брокеров можно переопределить через `RAG_KAFKA_BOOTSTRAP_SERVERS` и `RAG_REDIS_URL`. `RAG_QUEUE_NAME` задаёт имя для топика и стрима, по умолчанию `rag.jobs.v1`. При смене имени укажи его одинаково для `queue-init`, публикатора, диспетчера и воркеров. Команда `queue-init` пригодна для повторного запуска.

1. Веб вызывает `POST /api/conversations/{id}/jobs` с `{"request_id":"<UUID>","question":"..."}`. Ответ `202` содержит ID и начальный статус. Повтор с таким же ID и вопросом возвращает то же задание. Текст хранится в PostgreSQL.
2. Публикатор отправляет запись `job.ready` из outbox в Kafka. Диспетчер записывает ID задания в Redis и затем подтверждает Kafka offset. Из Kafka также доступны события `job.completed` и `job.failed`.
3. Воркер получает запись из Redis, берёт задачу с ограниченным по времени правом записи в PostgreSQL и вызывает RAG. Вопрос, ответ, цитаты и статус сохраняются одной транзакцией. Только после неё воркер подтверждает запись Redis. Следующий вопрос того же диалога становится доступен после завершения или исчерпания попыток предыдущего.
4. Веб опрашивает `GET /api/jobs/{id}`. После перезагрузки он подхватывает активные задания через `GET /api/conversations/{id}/jobs`. Если пользователь закрыл чат, задание продолжает обрабатываться.

Истёкшая попытка возвращается в очередь с задержкой, максимум три попытки. Публикатор также повторно сигнализирует готовые задания, если после публикации они долго остаются без воркера. Повторная доставка не дублирует сохранённый ответ. Повторный вызов модели возможен при падении между её ответом и фиксацией результата. В Langfuse обработку можно найти по `job_id` в наблюдении `rag.job`. Для проверки Kafka и Redis в тестах задай `RAG_TEST_QUEUE=1` вместе с `RAG_TEST_POSTGRES_DSN`.

## Как обновить корпус

Если тексты поменялись или на другой машине нужны актуальные пути к оригиналам, сначала обнови два экспорта. В CLI экспорт пока отсутствует. Этот пример использует существующие функции проекта.

```powershell
@'
from pathlib import Path

from rag.ingest.documents import iter_sections
from rag.ingest.export import write_sources
from rag.ingest.sources import source_from_section, source_from_ticket
from rag.ingest.tickets import iter_tickets

downloads = Path.home() / "Downloads"
tickets = downloads / "support_tickets_digit_dispatcher_200 (1).md"
documents = downloads / "zyfra_product_support_technical_reference (2).md"

write_sources(
    "data/processed/ticket_sources.jsonl",
    (source_from_ticket(ticket, tickets) for ticket in iter_tickets(tickets)),
)
write_sources(
    "data/processed/document_sources.jsonl",
    (source_from_section(section, documents) for section in iter_sections(documents)),
)
'@ | uv.exe run --no-sync python -
```

После экспорта последовательно выполни `build-index` и `build-hybrid`, затем перезапусти веб, чтобы он подключился к опубликованной версии. При изменении чанкинга или E5 с уже готовыми экспортами можно начинать с `build-index`.

## Оценка поиска, реранкер и Ragas

Для развёрнутой оценки используй [questions_v2.jsonl](data/eval/questions_v2.jsonl). Он содержит 100 контрольных вопросов, подготовленных по текущим экспортам корпуса. В нём 40 кейсов по решённым тикетам, 30 по документации, 15 с несколькими источниками, 10 на уточнение и 5 с ожидаемым отсутствием ответа. Старый набор из 14 вопросов остаётся набором для быстрой проверки.

`split=dev` отмечает 70 вопросов для настройки, `split=holdout` — 30 для итоговой проверки. Размеченные источники с одинаковым текстом находятся внутри одного сплита. Поле `type` отмечает тип кейса. `expected_clarification` служит для ручной проверки нужных уточнений. Текущий CLI читает все строки переданного файла. Для запуска отдельного сплита выдели строки по `split` в отдельный JSONL и передай путь через `--questions`.

```powershell
uv.exe run --no-sync rag eval --questions data/eval/questions_v2.jsonl
```

Метрики поиска считаются по 95 кейсам с размеченными источниками. Для многосоставных вопросов текущий `Hit@8` засчитывает попадание хотя бы одного источника. Полноту нахождения всех обязательных источников нужно оценить отдельно.

Для запросов документов к E5 и BM25 добавляется поиск по заголовкам и пути раздела из метаданных Qdrant. Новая индексация для этого не требуется. CLI-оценка использует этот же поиск и дедупликацию. Метрика `rrf` учитывает оба шага, а `rrf_raw` показывает исходное ранжирование E5 и BM25. Первый базовый прогон по обоим сплитам до внедрения дедупликации дал `Hit@8` 77/95 для E5, 66/95 для BM25 и 69/95 для RRF. У RRF хотя бы один размеченный источник попал в топ-20 в 83/95 кейсах. Прогон использовал текущий индекс и выполнялся без внешней модели.

Базовая оценка сравнивает E5, BM25 и гибридный поиск с ожидаемыми источниками. Она не обращается к внешнему LLM.

```powershell
uv.exe run --no-sync rag eval
```

`--rerank` добавляет сравнение с базовой RRF-выдачей и использует те же настройки модели и лимитов, что и помощник. Флаг оценки действует независимо от `RAG_RERANK_ENABLED`. `--ragas` добавляет генерацию и оценку ответов через API. Метрики включают `faithfulness` и, при наличии эталонного ответа, `factual_recall`. Результаты выводятся через CLI и могут быть сохранены в JSONL.

Для запуска с Ragas при работающем вебе можно создать отдельное окружение оценивания. Пример для второго терминала сохраняет результаты в новый файл, сохраняя прежний пилотный отчёт. Команда с обоими флагами вызывает настроенный API.

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'C:/Users/winda/AppData/Local/rag/eval-venv'
uv.exe run --locked --extra evaluation --env-file 'C:/Users/winda/AppData/Local/rag/polza.env' rag eval --rerank --ragas --case archive-screen --output data/eval/local-check.jsonl
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

Чтобы отправить трассы этой оценки в Langfuse, добавь `--extra observability` перед `--env-file`. Модульные тесты с подставными ответами запускаются командой `uv.exe run --no-sync python -m unittest discover -s tests -v`.

## Как продолжить в другом диалоге

В новом диалоге открой этот же проект и приложи этот файл. Для переноса на компьютер с нуля нужны исходный код, `pyproject.toml`, `uv.lock`, `compose.yaml` и два экспорта источников из `data/processed`. Для оценки также перенеси `data/eval/questions.jsonl`. Если переносишь оригинальные Markdown-файлы, можно заново экспортировать их для обновления путей к источникам. На новом компьютере создай собственный env-файл и собери коллекции по инструкции выше.

Сообщение для нового диалога

> Прочитай PROJECT_HANDOFF.md и сверь нужные детали с кодом. Это учебный RAG-проект, я хочу понимать каждый этап. Предложи несколько связанных шагов, укажи, какие файлы изменятся и как проверить результат. Дождись моего подтверждения перед правками. После изменений объясни, как их применить и проверить локально.
