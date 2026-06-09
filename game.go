package main

import (
	"errors"
	"sync"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

// [GO: STRUCT]
// Struct — як клас без методів, тільки дані.
// Методи додаються окремо через receiver: func (g *Game) Update()

// Game зберігає весь стан гри.
type Game struct {
	player   Pixel
	enemies  []Pixel // [GO: SLICE] — динамічний масив, як Array в JS
	gameOver bool

	// [GO: MEMORY SHARING]
	// sync.Mutex — "замок": mu.Lock() / mu.Unlock().
	// Зараз не активний — чекає коли додамо goroutines для ворогів.
	mu sync.Mutex

	// [GO: FIXED ARRAY]
	// Фіксований 2D масив — розмір відомий на компіляції, пам'ять одним блоком.
	boidMap [boidMapH][boidMapW]int

	tick       int     // лічильник кадрів
	difficulty float32 // множник складності (1.0 = старт)

	attackCooldown int // кадрів до наступного удару
	attackTimer    int // кадрів до кінця анімації кола
}

// [GO: SENTINEL ERROR]
// В Go немає exceptions — функції повертають error як звичайне значення.
// errors.New() створює унікальну помилку для порівняння: err == errExit.
var errExit = errors.New("exit")

// restart скидає стан гри до початкового.
func (g *Game) restart() {
	g.player.X = playerSpawn.X
	g.player.Y = playerSpawn.Y
	g.player.VX = 0
	g.player.VY = 0
	g.attackCooldown = 0
	g.attackTimer = 0
	g.enemies = newEnemies(enemyCount)
	g.gameOver = false
	g.tick = 0
	g.difficulty = 1.0
	currentPatternIdx = 0 // починаємо з першого патерну
	startBeat(1.0)
	for y := range g.boidMap {
		for x := range g.boidMap[y] {
			g.boidMap[y][x] = 0
		}
	}
}

// Update — головний цикл логіки, викликається ~60 разів на секунду.
func (g *Game) Update() error {
	if ebiten.IsKeyPressed(ebiten.KeyEscape) {
		return errExit
	}

	// P — вручну переключити патерн (для експериментів з ритмом)
	if inpututil.IsKeyJustPressed(ebiten.KeyP) {
		startBeat(g.difficulty)
	}

	if g.gameOver {
		if ebiten.IsKeyPressed(ebiten.KeyR) {
			g.restart()
		}
		return nil
	}

	// % — залишок від ділення (як в PHP/JS). tick%levelUpEvery==0 → новий рівень.
	g.tick++
	if g.tick%levelUpEvery == 0 && g.difficulty < maxDifficulty {
		g.difficulty += difficultyStep
		startBeat(g.difficulty) // темп зростає щорівня
	}

	if g.attackCooldown > 0 {
		g.attackCooldown--
	}
	if g.attackTimer > 0 {
		g.attackTimer--
	}

	g.handlePlayerInput()
	g.updatePlayer()
	g.playerAttack()
	g.updateBoidMap()
	g.calcAcceleration()
	g.updateEnemies()
	g.removeDeadEnemies()
	g.checkCollisions()
	return nil
}

// Layout — розмір логічного екрану (Ebiten масштабує під вікно).
func (g *Game) Layout(_, _ int) (int, int) {
	return screenWidth, screenHeight
}
