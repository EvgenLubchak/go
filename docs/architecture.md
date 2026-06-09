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
        → calcAcceleration()   (boids alignment + predator chase)
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

## Go Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| Pointer receiver `*Game` | All methods | Modify game state in-place |
| `for i := range` + `&slice[i]` | boids.go, combat.go | Avoid copy — modify enemy directly |
| Slice filter in-place | `removeDeadEnemies` | `alive := g.enemies[:0]` — no alloc |
| Zero value check | `pixel.go` | `cfg.Color.A == 0` detects Boid type |
| Sentinel error | `game.go` | `errors.New("exit")` for clean Ebiten exit |
| `sync.Mutex` | `game.go` | Reserved — will protect shared state when goroutines are added |
| `init()` | `main.go`, `sound.go` | One-time setup before `main()` |
| `var` over `const` for flags | `main.go` | Avoids linter "always true/false" warnings |
