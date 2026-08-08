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
| `brain_stack.go` | Шлях памʼяті **frame-stacking** (`useGRU=false`): `forwardQ`, `tdUpdate`, `stepStack` |
| `brain_gru.go` | Шлях памʼяті **GRU** (`useGRU=true`): `forwardGRU`, `tdUpdateSeq` (BPTT), `stepGRU` |
| `flowfield.go` | **Pathfinding**: BFS-хвиля від гравця, поле напрямків, `dirAt`/`distAt`, візуалізація (`V`) |
| `metrics.go` | Панель метрик навчання (`G`): reward/TD/maxQ по вуликах, частка часу наосліп, blind-chase (котлова й по агентах), catch-rate, ярлик конфігурації |
| `bench_test.go` | Безголовий стенд замірів: цикл гри без графіки, скриптований гравець, десятки прогонів на конфіг (`BOIDS_BENCH=1`) |
| `level.go` | Тайлова мапа рівня, спавни, `isWallAt`/`isWallRect` |
| `combat.go` | AABB collision, **[БІЙ] `resolveImpacts`** (шкода від closing speed + атрибуція), SPACE attack, смерть гравця, dead enemy removal |
| `render.go` | Малювання виду ЗВЕРХУ: тіло-восьмикутник, ворс, кінцівки, щупальце, кульки, HUD |
| `render3d.go` | Raycaster: вид від 1-ї особи (стіни + спрайти). Див. [raycaster.md](raycaster.md) |
| `sound.go` | Procedural 8-bit audio, drum patterns, BPM scaling |

---

## Темп: 120 ↔ 60 тіків (клавіша `T`)

Перемикання **безпечне за побудовою**, і причина варта запису: у грі **ніде немає
реального часу**. Усі числа в **КАДРАХ** — `dashWindup`, кулдауни, каденція атак,
горизонт γ, місткість буфера. Тож зниження TPS рівномірно сповільнює фізику, анімацію
**й навчання**: вони лишаються синхронними між собою.

Заміри теж не страждають — стенд крутиться безголово, без прив'язки до частоти.

> ⚠️ **Єдиний виняток — звук.** Біт рахується в РЕАЛЬНИХ секундах (`sixteenthSec`), тож
> BPM множиться на `tpsScale()`. Показово, що єдине місце з реальним часом — саме те,
> що не є симуляцією.

---

## Game Loop (Update → Draw, ~120 TPS)

```
Update():
  input → handlePlayerInput() | updatePrey()   ([SELF-PLAY] мозок-жертва замість клавіш)
        → updatePlayer()       (friction, max speed, стіни й межа)
        → updateFlowFields()   (multi-source BFS: по полю на кожну сторону)
        → playerAttack()       (SPACE → damage units in radius)
        → updateBoidMap()      (rebuild 2D grid of enemy positions)
        → calcAcceleration()   (boids + Brain.Step) ← parallel goroutines
        → trainBrains()        (навчання кожної УНІКАЛЬНОЇ мережі, однопотоково)
        → metrics.collect()
        → updateUnits()      (wander, burst, apply accel, bounce walls)
        → resolveImpacts()     ([БІЙ] шкода від удару на швидкості + атрибуція)
        → removeDeadUnits()  (filter slice in-place)
        → checkCollisions()    (HP гравця ≤ 0 → game over / respawn у self-play)

Draw():
  background → [flow-field] → units (сенсори, HP bar) → player → attack ring
             → HUD (HP, LVL, FPS) → [панель метрик]
```

---

## Key Data Structures

### Game
```go
type Game struct {
    player   Pixel
    units  []Pixel
    boidMap  [boidMapH][boidMapW]int  // 2D grid: 0=empty, i+1=enemy index
    tick       int
    difficulty float32                 // multiplier: 1.0 at start, grows per level
    mu         sync.Mutex              // reserved for future goroutines
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

**Активні зараз** (список `unitRoster` у `pixel.go`; кількість — поле `Count` у
кожному конфізі):

| Type | Колір | Сторона | Мозок / поведінка |
|------|-------|---------|-------------------|
| **Learner** | зелений | ворог | Реактивний переслідувач, POMDP. **Baseline** — навмисно не знає лабіринту |
| **Killer** | червоний | ворог | Знає лабіринт (**flow-field на вході**), швидший (1.6), 3 HP, **бойова нагорода** |
| **AllyChaser** | блакитний | гравець | Реактивний, полює на найближчого ворога, бойова нагорода |
| **AllyKiller** | фіолетовий | гравець | Flow-field до ВОРОГІВ — дзеркало червоного на твоєму боці |

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

Toggle: `soundEnabled = true/false` (var in main.go)

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
