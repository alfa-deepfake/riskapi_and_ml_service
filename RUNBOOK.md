# Runbook — ML service + RiskAPI stack

Полная инструкция: запуск, работа, остановка. Стек поднимает 4 контейнера:

| Сервис      | Внутри сети   | Порт хоста | Назначение                              |
|-------------|---------------|-----------|------------------------------------------|
| frontend    | `frontend:80` | `8080`    | браузерная консоль проверки              |
| ml-service  | `ml-service:8100` | `8100` | каскадный анти-дипфейк скоринг (ML)      |
| risk-api    | `risk-api:8000`   | `8000`* | приём и хранение статусов/результатов    |
| mongo       | `mongo:27017` | `27017`   | хранилище (`deepfake` db)                |

Поток данных: **браузер → ml-service → risk-api → Mongo**. Между собой сервисы
общаются по внутренним именам docker-сети, поэтому смена **хостовых** портов
пайплайн не ломает.

`*` На кластере `bc-a100-08` порты 8000 и 8001 заняты сторонними процессами,
поэтому risk-api там опубликован на **8002** через локальный `.env`
(`RISK_API_PORT=8002`, см. раздел «Порт занят»). `.env` не коммитится — иначе
он навязал бы свой порт всем хостам.

---

## 0. Предусловия (один раз)

- Установлен Docker + `docker compose` v2.
- Рядом лежит репозиторий `deepfake-riskapi` — стек собирает из него образ risk-api:

  ```
  alfa-deepfake/
  ├── deepfake-riskapi/          # сервис RiskAPI (собирается в образ risk-api)
  └── riskapi_and_ml_service/    # ML-сервис + compose (запускать отсюда)
  ```

Все команды ниже выполняются из каталога `riskapi_and_ml_service/`.

На кластере:
```bash
ssh -p 22010 master@62.183.4.208
cd /home/master/work/alfa-deepfake/riskapi_and_ml_service
```

---

## 1. Запуск

```bash
docker compose up -d --build      # собрать образы и поднять в фоне
```

Первая сборка идёт ~4 мин (тянет базовые образы и зависимости). Повторные
запуски используют кэш и стартуют за секунды.

Проверить, что всё поднялось:
```bash
docker compose ps
```
Все 4 контейнера должны быть `Up`.

Здоровье сервисов (подставьте порт risk-api — на кластере `8002`):
```bash
curl -s http://localhost:8100/health   # ml-service -> {"status":"ok",...}
curl -s http://localhost:8002/health   # risk-api   -> {"status":"ok"}  (ok = Mongo доступен)
```
Если risk-api вернул `{"detail":"MongoDB не запущен..."}` — Mongo ещё
инициализируется, подождите пару секунд и повторите.

### Прогнать пайплайн целиком (дымовой тест)
```bash
python3 scripts/smoke_stack.py
```
Ожидаемо: `"decision": "allow"`. Тест создаёт сессию в ML, шлёт evidence,
ML считает скор и отправляет статус+результат в risk-api → Mongo.

Убедиться, что результат реально записан в Mongo:
```bash
docker compose exec -T mongo mongosh deepfake --quiet \
  --eval 'db.scores.find().sort({updated_at:-1}).limit(1).pretty()'
```
Документ с вашим `check_id` в коллекции `scores` = цепочка замкнута.

---

## 2. Работа

**Фронтенд** (браузер): `http://localhost:8080`. Он ходит в ML по
`http://localhost:8100` (задано в `frontend/config.js`). При доступе с
локальной машины через SSH пробросьте порты:
```bash
ssh -p 22010 -L 8080:localhost:8080 -L 8100:localhost:8100 master@62.183.4.208
```

**Логи:**
```bash
docker compose logs -f                 # все сервисы
docker compose logs -f ml-service      # один сервис
docker compose logs -f risk-api
```

**Ручной вызов API** (порт risk-api — 8002 на кластере):
```bash
# ML: создать сессию
curl -s -X POST http://localhost:8100/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"uid":"u-1","check_id":"c-1","scenario":"video_call"}'

# RiskAPI: прочитать здоровье / записать результат вручную
curl -s -X POST http://localhost:8002/checks/c-1/result \
  -H 'Content-Type: application/json' \
  -d '{"uid":"u-1","score":{"decision":"allow","risk_score":0.1}}'
```

**Заглянуть в Mongo:**
```bash
docker compose exec mongo mongosh deepfake
# > db.scores.find().pretty()
# > db.statuses.find().pretty()
# > db.scores.countDocuments()
```

**Перезапустить один сервис** (например, после правки кода ml_service):
```bash
docker compose up -d --build ml-service
```

**Пересобрать всё с нуля** (после смены зависимостей/Dockerfile):
```bash
docker compose build --no-cache
docker compose up -d
```

---

## 3. Остановка

```bash
docker compose stop         # остановить контейнеры, данные и образы сохранить
docker compose start        # снова запустить остановленные
```

Полностью убрать контейнеры и сеть (том с данными Mongo сохранится):
```bash
docker compose down
```

Убрать вместе с данными Mongo (полный сброс):
```bash
docker compose down -v
```

---

## 4. Порт занят («address already in use»)

Симптом при `up`:
```
failed to bind host port 0.0.0.0:8000/tcp: address already in use
```

Внутренняя связка ml→risk-api от хостовых портов не зависит, поэтому risk-api
достаточно опубликовать на другом свободном порту. Кто держит порт:
```bash
ss -ltnp | grep :8000
```

Найти свободный порт и переопределить публикацию через локальный `.env`
(docker compose подхватывает его автоматически; хостовые порты в
`docker-compose.yml` заданы как `${RISK_API_PORT:-8000}` и т.п.):
```bash
echo "RISK_API_PORT=8002" >> .env   # 8002 — любой свободный порт хоста
docker compose up -d
```
После этого risk-api снаружи доступен на `http://localhost:8002`, а ml внутри
по-прежнему ходит на `risk-api:8000`. `.env` в `.gitignore` — он **локальный**,
чтобы не навязывать свой порт другим хостам (шаблон см. в `.env.example`).
На кластере создайте `.env` один раз (`echo RISK_API_PORT=8002 >> .env`).

Аналогично через `FRONTEND_PORT`/`MONGO_PORT` можно сдвинуть `frontend`/`mongo`.
**`ML_PORT` (8100) двигать нежелательно** — на него завязан `frontend/config.js`;
если двигаете, поправьте там `mlApiUrl` **и пересоберите образ frontend**
(config.js вшивается в образ на этапе build).

---

## 5. Частые проблемы

| Симптом | Причина / решение |
|---|---|
| risk-api `/health` = 503 «MongoDB не запущен» | Mongo ещё стартует — подождать; либо `docker compose logs mongo`. |
| `no such service: →` | В команду попал лишний текст. Запускать ровно `docker compose up -d --build`. |
| Порт занят при `up` | См. раздел 4. |
| Пересобрал ml_service, а изменений нет | `docker compose up -d --build ml-service` (без `--build` образ не пересобирается). |
| Аудио-чек отдаёт «model is not configured» | Чекпоинт WavLM не в git (380MB): скопировать на хост `scp .../best.pt <host>:.../riskapi_and_ml_service/models/audio/wavlm_all4_best.pt` — компоуз монтирует `./models/audio` в контейнер, достаточно рестарта без пересборки. |
| Аудио-чек = «phrase transcript is unavailable» | Образ собран до перехода на Faster-Whisper или его сборка не смогла скачать модель: проверьте доступ Docker к `huggingface.co`, затем выполните `docker compose up -d --build ml-service`. Снапшот `faster-whisper-medium` загружается и фиксируется в образе на этапе build; в рантайме сеть не нужна. |
| Первый запрос пульса долгий | Модель open-rppg строится ~1 мин; она греется в фоне при старте контейнера — дать сервису минуту после `up`. |
| Нужен GPU/тяжёлые модели | Базовый образ работает без них (адаптеры отдают «unavailable»). Для инференса моделей — доукомплектовать образ torch + чекпоинтами `neiro_model/`, см. комментарий в `Dockerfile` и `docker-compose.gpu.yml`. |
