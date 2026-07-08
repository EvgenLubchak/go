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

	// [СТИГМЕРГІЯ] Сітка «феромонів фрустрації»: де учні застрягають/б'ються об
	// стіни — накопичується слід, від якого рій відштовхується (і з часом тане).
	// Той самий патерн безпеки, що й boidMap: пишемо однопотоково (updateEnemies),
	// читаємо паралельно (calcAcceleration) — фази не перетинаються, гонок нема.
	frustration [boidMapH][boidMapW]float32

	tick       int     // лічильник кадрів
	difficulty float32 // множник складності (1.0 = старт)

	attackCooldown int // кадрів до наступного удару
	attackTimer    int // кадрів до кінця анімації кола

	// [RAYCASTER] Вид від першої особи (Wolfenstein-стиль). Симуляція лишається
	// 2D — змінюється ЛИШЕ камера/рендер. Перемикач: клавіша F.
	firstPerson bool    // false = вид зверху; true = від першої особи
	camAngle    float32 // напрямок камери (рад), слідує за напрямком руху гравця

	metrics Metrics // [МЕТРИКИ] крива навчання рою (клавіша G)
}

// [GO: SENTINEL ERROR]
// В Go немає exceptions — функції повертають error як звичайне значення.
// errors.New() створює унікальну помилку для порівняння: err == errExit.
var errExit = errors.New("exit")

// restart скидає стан гри до початкового.
func (g *Game) restart() {
	g.player.X = playerSpawn.X
	g.player.Y = playerSpawn.Y
	g.player.VelX = 0
	g.player.VelY = 0
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
	for y := range g.frustration {
		for x := range g.frustration[y] {
			g.frustration[y][x] = 0
		}
	}
}

// Update — головний цикл логіки, викликається ~60 разів на секунду.
func (g *Game) Update() error {
	if ebiten.IsKeyPressed(ebiten.KeyEscape) {
		g.saveBrains() // зберігаємо мозок перед виходом
		return errExit
	}

	// P — вручну переключити патерн (для експериментів з ритмом)
	if inpututil.IsKeyJustPressed(ebiten.KeyP) {
		startBeat(g.difficulty)
	}

	// F — перемикач виду: зверху ↔ від першої особи (raycaster)
	if inpututil.IsKeyJustPressed(ebiten.KeyF) {
		g.firstPerson = !g.firstPerson
	}

	// G — перемикач панелі метрик (крива навчання)
	if inpututil.IsKeyJustPressed(ebiten.KeyG) {
		showMetrics = !showMetrics
	}

	if g.gameOver {
		if ebiten.IsKeyPressed(ebiten.KeyR) {
			g.restart()
		}
		return nil
	}

	// % — залишок від ділення (як в PHP/JS). tick%levelUpEvery==0 → новий рівень.
	g.tick++
	if difficultyGrowth && g.tick%levelUpEvery == 0 && g.difficulty < maxDifficulty {
		g.difficulty += difficultyStep
		startBeat(g.difficulty) // темп зростає щорівня
	}

	if g.attackCooldown > 0 {
		g.attackCooldown--
	}
	if g.attackTimer > 0 {
		g.attackTimer--
	}

	if aiPlayer && g.player.Brain != nil {
		g.updatePrey() // [SELF-PLAY] гравцем керує мозок-жертва
	} else {
		g.handlePlayerInput()
	}
	g.updatePlayer()
	g.playerAttack()
	g.updateBoidMap()
	g.calcAcceleration()
	g.trainBrains() // [SHARED BRAIN] навчання мереж хижаків раз/кадр, ОДНОПОТОКОВО
	if aiPlayer && g.player.Brain != nil {
		g.player.Brain.net.train(qBatch) // [SELF-PLAY] тренуємо мозок-жертву
	}
	g.metrics.collect(g) // [МЕТРИКИ] збір показників навчання (однопотоково)
	g.updateEnemies()
	g.removeDeadEnemies()
	g.checkCollisions()
	return nil
}

// trainBrains — навчання мереж учнів РАЗ за кадр, ОДНОПОТОКОВО (після паралельної
// фази calcAcceleration). Кожну УНІКАЛЬНУ мережу тренуємо один раз: у режимі
// sharedBrain це одна спільна мережа, інакше — по одній на кожного учня.
//
// [GO: БЕЗПЕКА БЕЗ ЛОКУ] Запис ваг тут безпечний без мютекса, бо горутини
// воркер-пулу вже завершились (wg.Wait у calcAcceleration) — це та сама схема
// «всі читають паралельно → один пише однопотоково», що й для boidMap/феромонів.
func (g *Game) trainBrains() {
	seen := map[*Net]bool{}
	for i := range g.enemies {
		b := g.enemies[i].Brain
		if b == nil || b.net == nil || seen[b.net] {
			continue
		}
		seen[b.net] = true
		b.net.train(qBatch)
	}
}

// saveBrains зберігає ваги першого Learner-ворога у файл.
//
// [GO: JSON PERSISTENCE]
// Зберігаємо тільки першого — всі Learner-и починають з однакових ваг,
// тому зберігати кожного окремо не потрібно на цьому етапі.
func (g *Game) saveBrains() {
	for _, e := range g.enemies {
		if e.Brain != nil {
			SaveBrain(e.Brain)
			return
		}
	}
}

// Layout — розмір логічного екрану (Ebiten масштабує під вікно).
func (g *Game) Layout(_, _ int) (int, int) {
	return screenWidth, screenHeight
}
