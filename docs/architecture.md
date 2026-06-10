# Boids Game — Architecture

## File Structure

| File | Responsibility |
|------|---------------|
| `main.go` | Constants, font init, `main()` entry point |
| `game.go` | `Game` struct, `Update()` game loop, `restart()` |
| `pixel.go` | `Pixel` struct, `EnemyConfig`, 3 enemy type configs, `newEnemies()` |
| `player.go` | Player input handling, velocity/friction physics |
| `boids.go` | Boid AI: `updateBoidMap`, `calcAcceleration`, `updateEnemies` |
| `combat.go` | AABB collision, SPACE attack, HP damage, dead enemy removal |
| `render.go` | All drawing: pixels, HP bars, HUD, game over screen |
| `sound.go` | Procedural 8-bit audio, drum patterns, BPM scaling |

---

## Game Loop (Update → Draw, ~120 TPS)

```
Update():
  input → handlePlayerInput()
        → updatePlayer()       (friction, max speed, wrap-around)
        → playerAttack()       (SPACE → damage enemies in radius)
        → updateBoidMap()      (rebuild 2D grid of enemy positions)
        → calcAcceleration()   (boids alignment + predator chase) ← parallel goroutines
        → updateEnemies()      (wander, burst, apply accel, bounce walls)
        → removeDeadEnemies()  (filter slice in-place)
        → checkCollisions()    (enemy touches player → game over)

Draw():
  background → enemies (HP bar + label) → player → attack ring → HUD
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
    VX, VY     float32     // velocity vector
    AX, AY     float32     // acceleration (boids alignment + chase)
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

| Type | Label | Behavior |
|------|-------|----------|
| **Boid** | `FOE` | Flocks with neighbors, moderate aggression, color varies by Aggression |
| **Predator** | `PRD` | Red, 5HP, large detection (160px), powerful pounce ×14, ignores flock |
| **Speeder** | `SPD` | Yellow, 1HP, very fast (2.8), chaotic wander, frequent bursts |

Enemies cycle: `FOE, PRD, SPD, FOE, PRD, SPD, ...` (index % 3)

---

## Boids Algorithm

1. **boidMap** — 2D grid `[rows][cols]int`. Each cell stores `enemyIndex+1` (0=empty).
2. **Alignment** — each enemy scans `visionRadius` cells around itself, averages neighbor velocities, nudges toward that average × `AlignmentRate`.
3. **Chase (predator)** — if player is within `DetectionRange`, add acceleration toward player. Force multiplied by `pounce` factor that grows as distance shrinks.

```
pounce = (1 - dist/DetectionRange) * PounceMulti
AX += (dx/dist) * AggressionForce * Aggression * difficulty * (1 + pounce)
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
    ├── goroutine [0..n/8)    reads boidMap + vels snapshot → writes enemies[i].AX/AY
    ├── goroutine [n/8..n/4)  reads boidMap + vels snapshot → writes enemies[i].AX/AY
    ├── ...
    └── goroutine [7n/8..n)   reads boidMap + vels snapshot → writes enemies[i].AX/AY
  wg.Wait()               ← blocks until all workers done
  updateEnemies()         ← single-threaded (uses freshly written AX/AY)
```

**Snapshot pattern** — before parallelizing, VX/VY of all enemies are copied into
a local `[]vel` slice. Goroutines read from this snapshot (immutable), write only
to their own chunk of `enemies[i].AX/AY`. This avoids data races without any mutex.

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
| Snapshot before parallel work | `boids.go` | `[]vel` copy → race-free reads in goroutines |
| Closure argument capture | `boids.go` | `go func(start, end int)` avoids loop variable capture bug |
| `init()` | `main.go`, `sound.go`, `level.go` | One-time setup before `main()` |
| `var` over `const` for flags | `main.go` | Avoids linter "always true/false" warnings |
