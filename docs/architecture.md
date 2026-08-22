# Boids Game — Architecture

## File Structure

| File | Responsibility |
|------|---------------|
| `main.go` | Прапорці РЕЖИМІВ, темп (`gameTPS`), font init, `main()` |
| `tuning_world.go` | Ручки СВІТУ: розмір поля, тертя, стигмергія, темп рівнів, ритм |
| `tuning_combat.go` | Ручки БОЮ: удар, віддача, ривок, ухилення, рух гравця |
| `tuning_visual.go` | Ручки ВИГЛЯДУ: ворс, кульки, відросток, кінцівки, жало, тіло, фон |
| `game.go` | `Game` struct, `Update()` game loop, `restart()` |
| `pixel.go` | `Pixel` struct, `UnitConfig` (+`Count`), конфіги типів, **`unitRoster`** (склад поля), фракції, `newUnits()` |
| `player.go` | Player input handling, velocity/friction physics |
| `boids.go` | Boid AI: `updateBoidMap`, `calcAcceleration`, `updateUnits`, стигмергія (феромони) |
| `brain.go` | Мозок — **спільне ядро**: `Net`/`Brain`, індекси слотів, ε-greedy, `rewardFor`, whiskers, save/load, диспетчери `Step`/`train`. Див. [ai-brain.md](ai-brain.md) |
| `brain_stack.go` | Шлях памʼяті **frame-stacking** — за замовчуванням: `forwardQ`, `tdUpdate`, `stepStack` |
| `brain_gru.go` | ⚠️ **ПРИПАРКОВАНО** — шлях памʼяті GRU (лише явний `MemoryGRU`): `forwardGRU`, `tdUpdateSeq` (BPTT), `stepGRU`. Причини — у шапці файлу |
| `flowfield.go` | **Pathfinding**: BFS-хвиля від гравця, поле напрямків, `dirAt`/`distAt`, візуалізація (`V`) |
| `metrics.go` | Панель метрик навчання (`G`): reward/TD/maxQ по вуликах, частка часу наосліп, blind-chase (котлова й по агентах), catch-rate, ярлик конфігурації |
| `bench_test.go` | Безголовий стенд замірів: цикл гри без графіки, скриптований гравець, десятки прогонів на конфіг (`BOIDS_BENCH=1`) |
| `level.go` | Тайлова мапа рівня, спавни, `isWallAt`/`isWallRect` |
| `combat.go` | AABB collision, **[БІЙ] `resolveImpacts`** (шкода від closing speed + атрибуція), машина фаз РИВКА (SPACE: замах→ривок→відхід), `deathTransition`, респаун/видалення мертвих |
| `render.go` | Малювання виду ЗВЕРХУ: тіло-восьмикутник, ворс, кінцівки, щупальце, кульки, HUD |
| `render3d.go` | Raycaster: вид від 1-ї особи (стіни + спрайти). Див. [raycaster.md](raycaster.md) |
| `sound.go` | Procedural 8-bit audio, drum patterns, BPM scaling |
| `settings.go` | **Налаштування користувача** (settings.json): шари дефолт→файл→сесія, PATCH-семантика, лише відхилення, валідація |
| `panel.go` | **Панель налаштувань** (Tab): декларативна таблиця пунктів, гібридний ввід (миша/тачпад + клавіатура) |

---

## Камера: зум виду ЗВЕРХУ (клавіші `+` / `−`)

Зум — це **компроміс**, а не наближення: ближче видно деталі (форму тіла як прилад Q,
ворс, жало), але менше поля — тобто менше усвідомлення, звідки летить рій.

**Трансформація в точках виклику, а не масштабування картинки.** Найпростіший шлях —
намалювати світ в окреме зображення й розтягнути — при зумі 2× РОЗТЯГУЄ пікселі:
ворсинка 2px стає розмитою смугою. У нас усе векторне, тож координати переводяться ДО
малювання, і та сама ворсинка малюється як 4px у власній роздільності — різкою.

**Жодного особливого випадку.** Центр огляду відсікається до меж світу — камера не може
показати те, чого немає. Якби замість цього стояв `if zoom == 1`, він розійшовся б із
рештою коду на першій же правці.

> ⚠️ Раніше тут стояло: «при зумі 1.0 піввікна дорівнює півсвіту, тож трансформація
> тотожна». Це було правдою, поки світ дорівнював вікну. Тепер світ **3400×1960** проти
> вікна 1700×980, тож на 1.0 видно **чверть** карти й камера реально їздить. Окремого
> випадку це все одно не вимагає.

**Відсікання приходить задарма й діє завжди.** Вікно — чверть карти, тож 3/4 світу за
кадром **навіть на 1.0**; при 4× видно 425×245 з 3400×1960, тобто 1/64. Юніти й тайли
поза видимою областю просто не малюються.

**Типовий зум — 2.0, а не мінімум.** На 1.0 юніт займає 25 px, і дизайнерські дрібниці
(ворс, жало, форма тіла як прилад Q) просто не читаються.

Камера тягнеться за гравцем із тим самим **ВІДСТАВАННЯМ**, що вся анімація в грі
(`camFollow`): різкий стрибок різав би око, а лаг читається як інерція оператора.

---

## Малювання: два механізми, які не взаємозамінні

Найдорожчий урок оптимізації. У `vector` (ebiten 2.9) два різні шляхи до GPU:

| | як працює | ціна |
|---|---|---|
| `FillPath` / `StrokePath` | **накопичують** шляхи й віддають одним `DrawTriangles`, доки не зміняться antialias/blend/fillRule; растеризація через **атлас** | дорого за шлях, дешево за виклик |
| `FillRect` / `FillCircle` / `StrokeLine` | без AA йдуть **повз** цей стан, прямо в `DrawTriangles32` | дешево за фігуру, але **скидає** накопичений пакет |

**Правило, і воно НЕ «все через шлях»:** складна геометрія (ворс, кінцівки, відросток,
тіло, стіни) — у шлях, і бажано один шлях на групу; проста фігура (коло, прямокутник) —
лишається негайною.

> Ми перевірили це навпаки й **програли**: переведення кульок, долонь, смуг HP і моря на
> `FillPath` дало 30 FPS замість 63. CPU не змінився взагалі (0.441 проти 0.453 мс на
> 500 фігур) — уся ціна на боці GPU, бо 500 растеризацій шляху дорожчі за 500 пар
> трикутників.

**Що лишається правдою:** мішати їх упереміш — найгірше з двох. Тому `Draw` іде **трьома
проходами**: діагностичні оверлеї → шляхи всіх юнітів → прості фігури всіх. Один розрив
пакета замість 450.

Ціна видима й прийнята свідомо: юніт більше не малюється «цілком», тіла всіх лежать під
кульками всіх. Побічний виграш — смуга HP тепер завжди зверху, а раніше сусід,
намальований пізніше, перекривав її саме в купі.

---

## Згладжування: кадром, а не шляхом (панель Tab → «Графіка»)

**Ціна per-path AA росте з кількістю ШЛЯХІВ**, а не з площею малюнка: кожен шлях
малюється у трафарет **вісім разів** із субпіксельними зсувами, і кожен бере область
атласа **вдвічі ширшу**. У нас 10 шляхів на юніта → 50 видимих юнітів це 500 шляхів,
тобто **4000** малювань у трафарет проти 500.

Саме тому просадка приходила **в бою**: юніти збігаються в купу й разом потрапляють у
кадр, а поза боєм вони розмазані по світу, вчетверо більшому за вікно.

**Рішення — суперсемплінг** (`renderScale`, панель Tab, драбина 1 → 1.5 → 2 → 4). Світ
малюється у збільшений буфер і стискається на екран лінійною фільтрацією. Ціна **стала**:
одна текстура й одне стискання, байдуже скільки юнітів і чи вони бʼються. Ми міняємо
змінну ціну на фіксовану — а проблемою була саме нестабільність, не середнє.

Побічно згладжується **все**: кульки й долоні малюються трикутниками й повз per-path AA
проходили завжди.

Множення на масштаб живе всередині `cam.px/py/s` — камера і так єдиний місток зі світу
на екран, тож тридцять точок малювання про буфер не знають. HUD і текст малюються
**після** стискання, у рідній роздільності, тож лишаються різкими.

> **Чому драбина закінчується на ×4.** Були ще ×8 і ×12. На ×12 гра **впала**, і впала
> повз `recover`: буфер 20400×11760 це майже гігабайт, і невдача такого виділення
> приходить не з Go, а з драйвера. Звідси правило: у драбині лишаються **лише перевірені
> в грі** щаблі.
>
> Вище ×2 картинка майже не кращає: фільтр читає 2×2 тексели, тож із 16 пікселів буфера
> при ×4 він бачить чотири (решту домішують мипмапи), а платня росте як **квадрат**.
> ×4 лишений як **прилад**: усе інше в кадрі незмінне, тож просадка між ×2 і ×4 міряє
> рівно ціну пікселя.

**Типово:** per-path AA вимкнено, `renderScale = 1.5`. Обидва перемикаються на
панелі налаштувань (Tab, гібрид: клік/тачпад або стрілки+Enter) і персистяться в
`settings.json` — ЛИШЕ відхилення від дефолтів, PATCH-семантика (відсутній ключ =
дефолт коду), значення SS валідуються проти драбини. Деталі — `settings.go`.

---

## Заморозка кадрів на ударі (hitstop)

Удар триває один тік, і око не встигає його зареєструвати — воно бачить наслідок
(відкид, спалах), але не сам момент. Пауза тримає на екрані кадр максимальної інформації
й дає удару **вагу**: він коштує часу, а не лише здоровʼя.

**Лише гравець.** При `impactInvuln = 45` і півсотні юнітів бійня фракцій дає десятки
влучань за секунду — заморозка на кожному спинила б гру назавжди.

**Асиметрія:** отримав удар — довше (3 кадри), завдав — коротше (2), кулдаун 5. Це нижче
за класичні 4–8 недарма: у нас ривок і так відбирає керування, тож заморозка додається
до вже наявного очікування.

Три рішення, які лишили б тихі вади, якби пішли інакше:

- `applyImpactDamage` повертає `bool` — морозимо на **подію**, а не на спробу. Удар у
  невразливого це промах;
- беремо **більшу** з двох, а не суму: в одному тіку можна і вдарити, і дістати;
- місце в `Update` — **після всіх клавіш** (морозимо світ, а не введення: інакше
  натискання за ці кадри зникло б повз `IsKeyJustPressed`), але **до** `cam.follow`
  (інакше камера їхала б далі, і це читалось би як підвисання), і **окремо від паузи**
  (їй `cam.follow` потрібен — саме він відсікає огляд до меж світу).

`TestHitstopDoesNotReachTheBench` стереже найдорожче: шов проходить по `Update`, тож
`tickHeadless` заморозки не бачить і всі базові лінії лишаються порівнянними.

---

## Темп: 60 ↔ 120 тіків (панель Tab, типово **60**)

Перемикання **безпечне за побудовою**, і причина варта запису: у грі **ніде немає
реального часу**. Усі числа в **КАДРАХ** — `dashWindup`, кулдауни, каденція атак,
горизонт γ, місткість буфера. Тож зниження TPS рівномірно сповільнює фізику, анімацію
**й навчання**: вони лишаються синхронними між собою.

Заміри теж не страждають — стенд крутиться безголово, без прив'язки до частоти.

> ⚠️ **Єдиний виняток — звук.** Біт рахується в РЕАЛЬНИХ секундах (`sixteenthSec`), тож
> BPM множиться на `tpsScale()`. Показово, що єдине місце з реальним часом — саме те,
> що не є симуляцією.

---

## Game Loop (Update → Draw, типово 60 TPS)

```
Update():
  клавіші режимів → [hitstop: світ стоїть] → cam.follow → [пауза]
  input → playerDashInput() + handlePlayerInput() | updatePrey()  ([SELF-PLAY] мозок-жертва)
        → updatePlayer()       (friction, ривок, стіни й межа; тікають таймери ухилення гравця)
        → updateFlowFields()   (multi-source BFS: по полю на кожну сторону, троттлинг)
        → updateBoidMap()      (rebuild 2D grid of unit positions)
        → calcAcceleration()   (boids + Brain.Step; ухилення = лише НАМІР) ← parallel goroutines
        → trainBrains()        (навчання кожної УНІКАЛЬНОЇ мережі, однопотоково)
        → metrics.collect()
        → updateUnits()        (відкладені кидки, таймери, apply accel, bounce walls)
        → resolveImpacts()     ([БІЙ] шкода від closing speed + атрибуція + hitstop-запит)
        → pushOffPlayer()      (юніти не стоять УСЕРЕДИНІ гравця; після шкоди — порядок критичний)
        → handleDeadUnits()    (deathTransition → респаун на пост або видалення)
        → checkCollisions()    (HP гравця ≤ 0 → game over / respawn у self-play)

Draw():
  море → стіни (один шлях) → [феромони] → [flow-field] → ШЛЯХИ всіх юнітів
       → ПРОСТІ ФІГУРИ всіх (очі, HP, зуби одним викликом) → КУЛЬКИ → ривок
       → HUD (HP, LVL, FPS/TPS, зум/SS) → [панель метрик]
```

---

## Key Data Structures

### Game
```go
type Game struct {
    player      Pixel
    units       []Pixel
    hive        map[string]*Net           // реєстр мереж: файл ваг → Net (переживає вимирання типу)
    boidMap     [boidMapH][boidMapW]int   // 2D grid: 0=empty, i+1=unit index
    frustration [boidMapH][boidMapW]float32 // феромони фрустрації (стигмергія)
    flowToPlayerSide, flowToEnemySide FlowField // маршрути крізь лабіринт, по полю на сторону
    tick        int
    difficulty  float32 // multiplier: 1.0 at start
    hitstop     int     // заморозка кадрів після удару за участю гравця
    paused      bool
    metrics     Metrics
}
```

### Pixel (player or enemy)
```go
type Pixel struct {
    X, Y, VelX, VelY, AccX, AccY float32
    HP, MaxHP int
    HitTimer  int          // біле блимання N кадрів після удару
    Cfg       UnitConfig   // конфіг типу (порожній для гравця)

    // [БІЙ] Таймери, що керують і фізикою, і вразливістю
    InvulnTimer, KnockTimer     int
    DodgeTimer, DodgeCooldown   int   // девʼята дія: ухилення
    KnockResist, KnockRecoil    float32

    // [АНІМАЦІЯ] Усе у СВІТОВИХ координатах — саме тому воно й відстає
    Fur   [furStrands][furJoints][2]float32
    Limbs [limbCount][2]float32
    Tent  [tentJoints][2]float32
    Balls [ballCount][2]float32
    BodyScale, BodyVel float32          // пружина розміру
    StingTimer int                      // постріл щупальця
    StingDirX, StingDirY float32
}
```

---

## Анімація: пʼять ефектів, ОДИН механізм

**Жодна анімація в грі не намальована.** Уся виникає з одного правила — **ВІДСТАВАННЯ**:
точка щокадру підтягується до своєї цілі лише на частку шляху й принципово не встигає.

```go
точка += (ціль − точка) × жорсткість
```

| ефект | ціль точки | що дає відставання |
|---|---|---|
| **ворс** | попередній суглоб + сегмент У ТОМУ Ж напрямку | у спокої стирчить, на русі хльоскає |
| **кульки** | точка під тілом | теліпаються на поворотах |
| **щупальце** | попередній суглоб + сегмент УНИЗ | обвисає, хвиля біжить згори вниз |
| **кінцівки** | кут корпуса + напрямок звисання | руки й ноги бовтаються |
| **пружина тіла** | розмір, що падає зі швидкістю | пульс і перельот при зупинці |

Дві відмінності, які легко загубити при узагальненні:

- **ворс** цілиться «продовжити НАПРЯМОК» → у спокої **прямий**;
- **щупальце** цілиться «сегмент УНИЗ» → у спокої **обвисає**.

Одна формула на обидва зламала б один із них молча.

### Стани поверх того ж механізму

Наїжачення при фрустрації й **жало** — це не окремі анімації, а **множники довжини й
жорсткості на СТАН**. Повернення програмувати не треба: щойно стан спаде, ціль знову
стара, і те саме відставання поверне точку з розмаху.

### Кріплення їдуть за тілом

Усі корені множаться на `bodyScaleOf(p)` — поточний масштаб пружини. Без цього тіло
стискається на швидкості (`BodyScale ~0.78`), а кріплення лишаються на сталій
півсторони, і щупальце візуально **відривається** від корпуса на ~2.2px.

### Вартість: 128 → 13 викликів малювання на юніта

Кожен сегмент окремим `StrokeLine` давав 112 викликів на самий лише ворс при
`furJoints = 14` — 88% усієї вартості. Тепер сегменти збираються у **спільні шляхи** і
малюються одним `StrokePath`:

| | було | стало |
|---|---|---|
| ворс | 112 | **3** (групи за товщиною) |
| кінцівки | 8 | 5 |
| щупальце | 5 | 2 |
| **разом** | **128** | **13** |

Товщина в `StrokeOptions` одна на шлях — тому групи й потрібні. Три сходинки на
волосині `2→1px` оком не відрізнити від плавного звуження.

### UnitConfig — per-type behavior
Each enemy type carries its own behavioral parameters instead of using global constants.

```go
type UnitConfig struct {
    WanderStrength, AlignmentRate float32
    MaxSpeed, AggressionForce     float32
    BurstChance, BurstForce       float32
    DetectionRange, PounceMulti   float32
    MaxHP  int
    Color  color.RGBA  // A==0 → color determined by Aggression field
    Label  string
}
```

---

## Enemy Types

**Склад поля** — список `unitRoster` у `pixel.go`; кількість — поле `Count` у
кожному конфізі (`Count: 0` = тип у ростері, але вимкнений):

| Type | Колір | Сторона | Мозок / поведінка |
|------|-------|---------|-------------------|
| **Learner** | зелений | ворог | Реактивний переслідувач, POMDP. **Baseline** — навмисно не знає лабіринту |
| **Killer** | червоний | ворог | Знає лабіринт (**flow-field на вході**), швидший (1.6), **бойова нагорода** |
| **AllyChaser** | блакитний | гравець | Реактивний, полює на найближчого ворога, бойова нагорода |
| **AllyKiller** | фіолетовий | гравець | Flow-field до ВОРОГІВ — дзеркало червоного на твоєму боці |
| **Warden** | золотий | ворог | СТРАЖНИК: лише бойова нагорода (`CombatOnly`), тримає місце, γ0.99, буфер 65536 |
| **Boss** | малиновий | ворог | ОДИНАК з власною мережею (зараз `Count: 0` — вимкнений) |
| **Hunter** | помаранчевий | ворог | бій + наосліп + рухомий (зараз `Count: 0` — вимкнений) |

Кожен тип учиться **незалежно**: свій вулик і свій файл ваг (`WeightsFile` у конфізі),
тож зміни для одного не чіпають інших. Склад поля — `unitRoster` + `Count` у конфігах.
Деталі — [ai-brain.md](ai-brain.md).

**Скриптовані типи** (без мозку) лишились у коді як приклади конфігурації, але не в
складі поля: `Boid` (флокується), `Predator` (повільний, сильний кидок), `Speeder`
(швидкий і крихкий), `HP`, `Group`. Щоб повернути в бій — дописати в `unitRoster`
і виставити `Count`.

---

## Boids Algorithm

1. **boidMap** — 2D grid `[rows][cols]int`. Each cell stores `enemyIndex+1` (0=empty).
2. **Alignment** — each enemy scans `visionRadius` cells around itself, averages neighbor velocities, nudges toward that average × `AlignmentRate`.
3. **Chase (predator)** — if player is within `DetectionRange`, add acceleration toward player. Force multiplied by `pounce` factor that grows as distance shrinks.

```
pounce = (1 - dist/DetectionRange) * PounceMulti
AccX += (dx/dist) * AggressionForce * Aggression * difficulty * (1 + pounce)
```

---

## Difficulty Scaling

Every `levelUpEvery` ticks: `difficulty += difficultyStep`

Affects:
- Enemy max speed: `Cfg.MaxSpeed * difficulty`
- Chase force: `AggressionForce * difficulty`
- Audio BPM: `baseBPM + (difficulty-1) * bpmPerDifficulty`
- Player max speed: `playerBaseSpeed * sqrt(difficulty)`

---

## Audio System

Procedural 8-bit square wave. Pattern cycles through 4 drum patterns on each level up.

```
baseBPM=90 → maxBPM=5000, scaled by difficulty
16-step sequencer: kick / snare / hi-hat per step
Square wave: sin(t) ≥ 0 → +amplitude, else −amplitude (retro sound)
Loop: audio.NewInfiniteLoop — auto-rewinds at end of measure
```

Перемикачі: панель Tab → «Звук» (персиститься в settings.json) і «Ритм — барабанний патерн» (стан сесії, не персиститься)

---

## Concurrency — Worker Pool in calcAcceleration

`calcAcceleration` is the most expensive function: O(n × visionRadius²) per frame.
With 3000+ units it becomes the bottleneck → parallelized across CPU cores.

```
Main goroutine:
  updateBoidMap()         ← single-threaded (builds shared read-only grid)
  calcAcceleration()      ← spawns NumCPU workers via sync.WaitGroup
    ├── goroutine [0..n/8)    reads boidMap + snaps snapshot → writes units[i].AccX/AccY
    ├── goroutine [n/8..n/4)  reads boidMap + snaps snapshot → writes units[i].AccX/AccY
    ├── ...
    └── goroutine [7n/8..n)   reads boidMap + snaps snapshot → writes units[i].AccX/AccY
  wg.Wait()               ← blocks until all workers done
  updateUnits()         ← single-threaded (uses freshly written AccX/AccY)
```

**Snapshot pattern** — before parallelizing, VelX/VelY of all units are copied into
a local `[]snap` slice. Goroutines read from this snapshot (immutable), write only
to their own chunk of `units[i].AccX/AccY`. This avoids data races without any mutex.

**Why not one goroutine per enemy?** Goroutine creation has overhead (~1µs).
With 3752 units × 120 FPS = 450k goroutine launches/sec — marginal.
Worker pool (NumCPU goroutines) amortizes this: each goroutine processes n/CPU units.

---

## Go Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| Pointer receiver `*Game` | All methods | Modify game state in-place |
| `for i := range` + `&slice[i]` | boids.go, combat.go | Avoid copy — modify enemy directly |
| Slice filter in-place | `removeDeadUnits` | `alive := g.units[:0]` — no alloc |
| Zero value check | `pixel.go` | `cfg.Color.A == 0` detects Boid type |
| Sentinel error | `game.go` | `errors.New("exit")` for clean Ebiten exit |
| `sync.WaitGroup` | `boids.go` | Synchronizes worker goroutines in calcAcceleration |
| Snapshot before parallel work | `boids.go` | `[]snap` copy → race-free reads in goroutines |
| Closure argument capture | `boids.go` | `go func(start, end int)` avoids loop variable capture bug |
| `init()` | `main.go`, `sound.go`, `level.go` | One-time setup before `main()` |
| `var` over `const` for flags | `main.go` | Avoids linter "always true/false" warnings |
