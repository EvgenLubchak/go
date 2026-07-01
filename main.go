package main

import (
	"bytes"
	"image/color"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	etext "github.com/hajimehoshi/ebiten/v2/text/v2"
	"golang.org/x/image/font/gofont/goregular"
)

// Всі константи в одному місці — легко знайти і змінити.
// В Go константи пакету видимі у всіх файлах цього пакету.
const (
	screenWidth  = 1680
	screenHeight = 960
	pixelSize    = 20
	enemyCount   = 100

	// Глобальна фізика — однакова для всіх типів ворогів.
	// Поведінка (швидкість, агресія, burst) — в EnemyConfig у pixel.go.
	damping             = 1     // множник швидкості щокадру (1 = без тертя, 0.9 = гальмує)
	visionRadius        = 3     // радіус огляду в клітинках boidMap
	showDetectionCircle = false // показувати радіус огляду ворогів (true/false)

	// [СТИГМЕРГІЯ] Феромони фрустрації — рій лишає сліди в глухих місцях і обходить їх.
	frustrationDeposit = 1     // слід за ОДНЕ застрягання (тепер рідко → більший внесок)
	frustrationDecay   = 0.970 // затухання сліду щокадру (місце поступово «забувається»)
	frustrationRepel   = 0.05  // сила відштовхування від слідів
	frustrationRadius  = 3     // радіус сканування слідів навколо агента (клітинки)

	attackRadius      = 120 // радіус удару в пікселях
	attackDamage      = 1   // пошкодження за один удар
	attackCooldownMax = 10  // кадрів між ударами
	attackDuration    = 10  // кадрів відображення кола атаки
	hitFlashDuration  = 7   // кадрів білого миготіння після удару

	playerAccel     = 0.45 // прискорення при натисканні клавіші
	playerFriction  = 0.90 // тертя щокадру
	playerBaseSpeed = 5.0  // макс швидкість гравця на рівні 1
	playerTurnSpeed = 0.03 // [RAYCASTER] швидкість повороту камери (рад/кадр) у виді 1-ї особи

	labelFontSize    = 6.0  // розмір шрифту мітки на пікселі
	gameOverFontSize = 20.0 // розмір шрифту екрану GAME OVER

	levelUpEvery   = 60 * 60 // кожні 10 секунд (60fps × 5)
	difficultyStep = 2       // наскільки зростає складність за рівень
	maxDifficulty  = 250.0   // стеля складності

	// Аудіо: темп зростає разом зі складністю
	baseBPM          = 90.0   // BPM на рівні 1
	maxBPM           = 5000.0 // стеля темпу
	bpmPerDifficulty = 24.0   // скільки BPM додається за одиницю difficulty

	boidMapW = screenWidth / pixelSize  // клітинок по горизонталі
	boidMapH = screenHeight / pixelSize // клітинок по вертикалі
)

// [GO: PACKAGE-LEVEL VAR + INIT]
// var на рівні пакету — існує весь час роботи програми.
// init() викликається автоматично до main() — завантажуємо шрифт один раз.
// soundEnabled можна змінити під час роботи програми (на відміну від const).
// [GO: VAR vs CONST] — var дозволяє умовні перевірки без попереджень лінтера.
var (
	fontFaceSource      *etext.GoTextFaceSource
	soundEnabled        = false // false — вимкнути фоновий ритм
	difficultyGrowth    = false // false — складність не росте (для тренування AI)
	showWhiskers        = false // показувати сенсори стін і обрану дію Learner-а
	showMetrics         = true  // показувати панель метрик навчання (крива reward/TD) — клавіша G
	showFrustration     = false // показувати теплову карту феромонів фрустрації
	pheromonesEnabled   = true  // вмикає феромони фрустрації (стигмергію); false = чистий Q-learning без слідів
	epsilonDecayEnabled = false // ε: false = постійна (qEpsilonConst); true = автоспад max→min
	sharedBrain         = true  // true = всі учні ділять ОДНУ мережу (вулик-розум); false = кожен свою
)

func init() {
	src, err := etext.NewGoTextFaceSource(bytes.NewReader(goregular.TTF))
	if err != nil {
		log.Fatal("font load:", err)
	}
	fontFaceSource = src
}

func main() {
	game := &Game{
		difficulty: 1.0,
		player: Pixel{
			X:     playerSpawn.X,
			Y:     playerSpawn.Y,
			Color: color.RGBA{R: 0, G: 255, B: 180, A: 255},
			Label: "Y}{IJIEC",
		},
		enemies: newEnemies(enemyCount),
	}

	startBeat(1.0) // запускаємо аудіо перед стартом гри

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Boids")
	ebiten.SetTPS(120)

	if err := ebiten.RunGame(game); err != nil && err != errExit {
		log.Fatal(err)
	}
}
