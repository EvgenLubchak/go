# Boids Game — Architecture

## File Structure

| File | Responsibility |
|------|---------------|
| `main.go` | Constants, font init, `main()` entry point |
| `game.go` | `Game` struct, `Update()` game loop, `restart()` |
| `pixel.go` | `Pixel` struct, `EnemyConfig` (+`Count`), конфіги типів, **`enemyRoster`** (склад поля), фракції, `newEnemies()` |
| `player.go` | Player input handling, velocity/friction physics |
| `boids.go` | Boid AI: `updateBoidMap`, `calcAcceleration`, `updateEnemies`, стигмергія (феромони) |
| `brain.go` | Мозок — **спільне ядро**: `Net`/`Brain`, індекси слотів, ε-greedy, `rewardFor`, whiskers, save/load, диспетчери `Step`/`train`. Див. [ai-brain.md](ai-brain.md) |
| `brain_stack.go` | Шлях памʼяті **frame-stacking** (`useGRU=false`): `forwardQ`, `tdUpdate`, `stepStack` |
| `brain_gru.go` | Шлях памʼяті **GRU** (`useGRU=true`): `forwardGRU`, `tdUpdateSeq` (BPTT), `stepGRU` |
| `flowfield.go` | **Pathfinding**: BFS-хвиля від гравця, поле напрямків, `dirAt`/`distAt`, візуалізація (`V`) |
| `metrics.go` | Панель метрик навчання (`G`): reward/TD/maxQ, blind-chase, catch-rate |
| `level.go` | Тайлова мапа рівня, спавни, `isWallAt`/`isWallRect` |
| `combat.go` | AABB collision, **[БІЙ] `resolveImpacts`** (шкода від closing speed + атрибуція), SPACE attack, смерть гравця, dead enemy removal |
| `render.go` | Малювання виду ЗВЕРХУ: pixels, HP bars, HUD, game over |
| `render3d.go` | Raycaster: вид від 1-ї особи (стіни + спрайти). Див. [raycaster.md](raycaster.md) |
| `sound.go` | Procedural 8-bit audio, drum patterns, BPM scaling |

---

## Game Loop (Update → Draw, ~120 TPS)

```
Update():
  input → handlePlayerInput() | updatePrey()   ([SELF-PLAY] мозок-жертва замість клавіш)
        → updatePlayer()       (friction, max speed, стіни й межа)
        → updateFlowField()    (BFS від гравця — лише коли змінив клітинку)
        → playerAttack()       (SPACE → damage enemies in radius)
        → updateBoidMap()      (rebuild 2D grid of enemy positions)
        → calcAcceleration()   (boids + Brain.Step) ← parallel goroutines
        → trainBrains()        (навчання кожної УНІКАЛЬНОЇ мережі, однопотоково)
        → metrics.collect()
        → updateEnemies()      (wander, burst, apply accel, bounce walls)
        → resolveImpacts()     ([БІЙ] шкода від удару на швидкості + атрибуція)
        → removeDeadEnemies()  (filter slice in-place)
        → checkCollisions()    (HP гравця ≤ 0 → game over / respawn у self-play)

Draw():
  background → [flow-field] → enemies (сенсори, HP bar) → player → attack ring
             → HUD (HP, LVL, FPS) → [панель метрик]
```

---

## Key Data Structures

### Game
```go
type Game struct {
    player   Pixel
    enemies  []Pixel
    boidMap  [boidMapH][boidMapW]int  // 2D grid: 0=empty, i+1=enemy index
    tick       int
    difficulty float32                 // multiplier: 1.0 at start, grows per level
    mu         sync.Mutex              // reserved for future goroutines
}
```

### Pixel (player or enemy)
```go
type Pixel struct {
    X, Y       float32
    VelX, VelY     float32     // velocity vector
    AccX, AccY     float32     // acceleration (boids alignment + chase)
    Aggression float32     // 0..1: Boid=random, Predator/Speeder=1.0
    HP, MaxHP  int
    HitTimer   int         // flash white for N frames after hit
    Cfg        EnemyConfig // behavior config (empty for player)
}
```

### EnemyConfig — per-type behavior
Each enemy type carries its own behavioral parameters instead of using global constants.

```go
type EnemyConfig struct {
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

**Активні зараз** (список `enemyRoster` у `pixel.go`; кількість — поле `Count` у
кожному конфізі):

| Type | Колір | Мозок | Поведінка |
|------|-------|-------|-----------|
| **Learner** | зелений | вулик `brain_weights.json` | Реактивний переслідувач, POMDP (`localSight`), памʼять GRU. **Baseline** — навмисно не знає лабіринту |
| **Killer** | червоний | вулик `killer_weights.json` | Знає лабіринт (**flow-field на вході**), всевидющий, швидший (1.6), 3 HP, **бойова нагорода** |

Обидва типи вчаться **незалежно** (окремі мережі + окремі файли ваг), тож зміни для
вбивці не чіпають налаштований рій. Деталі — [ai-brain.md](ai-brain.md).

**Скриптовані типи** (без мозку) лишились у коді як приклади конфігурації, але не в
складі поля: `Boid` (флокується), `Predator` (повільний, сильний кидок), `Speeder`
(швидкий і крихкий), `HP`, `Group`. Щоб повернути в бій — дописати в `enemyRoster`
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
With 3000+ enemies it becomes the bottleneck → parallelized across CPU cores.

```
Main goroutine:
  updateBoidMap()         ← single-threaded (builds shared read-only grid)
  calcAcceleration()      ← spawns NumCPU workers via sync.WaitGroup
    ├── goroutine [0..n/8)    reads boidMap + snaps snapshot → writes enemies[i].AccX/AccY
    ├── goroutine [n/8..n/4)  reads boidMap + snaps snapshot → writes enemies[i].AccX/AccY
    ├── ...
    └── goroutine [7n/8..n)   reads boidMap + snaps snapshot → writes enemies[i].AccX/AccY
  wg.Wait()               ← blocks until all workers done
  updateEnemies()         ← single-threaded (uses freshly written AccX/AccY)
```

**Snapshot pattern** — before parallelizing, VelX/VelY of all enemies are copied into
a local `[]snap` slice. Goroutines read from this snapshot (immutable), write only
to their own chunk of `enemies[i].AccX/AccY`. This avoids data races without any mutex.

**Why not one goroutine per enemy?** Goroutine creation has overhead (~1µs).
With 3752 enemies × 120 FPS = 450k goroutine launches/sec — marginal.
Worker pool (NumCPU goroutines) amortizes this: each goroutine processes n/CPU enemies.

---

## Go Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| Pointer receiver `*Game` | All methods | Modify game state in-place |
| `for i := range` + `&slice[i]` | boids.go, combat.go | Avoid copy — modify enemy directly |
| Slice filter in-place | `removeDeadEnemies` | `alive := g.enemies[:0]` — no alloc |
| Zero value check | `pixel.go` | `cfg.Color.A == 0` detects Boid type |
| Sentinel error | `game.go` | `errors.New("exit")` for clean Ebiten exit |
| `sync.WaitGroup` | `boids.go` | Synchronizes worker goroutines in calcAcceleration |
| Snapshot before parallel work | `boids.go` | `[]snap` copy → race-free reads in goroutines |
| Closure argument capture | `boids.go` | `go func(start, end int)` avoids loop variable capture bug |
| `init()` | `main.go`, `sound.go`, `level.go` | One-time setup before `main()` |
| `var` over `const` for flags | `main.go` | Avoids linter "always true/false" warnings |
