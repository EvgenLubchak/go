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
	screenWidth  = 1700
	screenHeight = 980
	pixelSize    = 25
	// [СКЛАД ПОЛЯ] Кількість ворогів більше НЕ тут: кожен тип несе своє поле Count,
	// а хто виходить на поле — список unitRoster (pixel.go). Так додати новий тип
	// = один рядок, і неможливо мовчки лишитись без переслідувачів.

	// Глобальна фізика — однакова для всіх типів ворогів.
	// Поведінка (швидкість, агресія, burst) — в UnitConfig у pixel.go.
	damping             = 1     // множник швидкості щокадру (1 = без тертя, 0.9 = гальмує)
	visionRadius        = 3     // радіус огляду в клітинках boidMap
	showDetectionCircle = false // показувати радіус огляду ворогів (true/false)

	// [СТИГМЕРГІЯ] Феромони фрустрації — рій лишає сліди в глухих місцях і обходить їх.
	frustrationDeposit = 1     // слід за ОДНЕ застрягання (тепер рідко → більший внесок)
	frustrationDecay   = 0.970 // затухання сліду щокадру (місце поступово «забувається»)
	frustrationRepel   = 0.05  // сила відштовхування від слідів
	frustrationRadius  = 3     // радіус сканування слідів навколо агента (клітинки)

	// [БІЙ] Шкода від УДАРУ НА ШВИДКОСТІ («кидок кобри»).
	// Шкодить не сам дотик, а ЗБЛИЖЕННЯ на швидкості: беремо проєкцію швидкості
	// на напрямок до цілі (closing speed). Асиметрично — шкоду завдає той, хто
	// летить У іншого. Тому: повільно зіштовхнулись = нічого; налетів = вкусив;
	// лоб-у-лоб = обидва отримали. «Кидок і відступ» виникає САМ (після удару
	// швидкість витрачена → ти вразливий), його не треба програмувати.
	//
	// Поріг — ЧАСТКА власної максимальної швидкості, а не абсолют: гравець
	// (5.0 px/кадр) у ~6 разів швидший за учня (0.8), тож абсолютний поріг зробив
	// би гравця танком, а ворогів — безпечними. Плюс абсолютна підлога, щоб
	// повільні типи не вбивали «повзком».
	impactSpeedFrac = 0.6 // ≥60% власного максимуму в бік цілі → удар
	impactMinSpeed  = 0.4 // абсолютна підлога швидкості удару (px/кадр)
	impactDamage    = 1   // шкода за один удар
	impactInvuln    = 45  // кадрів невразливості після отриманого удару
	playerMaxHP     = 10  // здоровʼя гравця (більше не вмирає з одного дотику)

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

	// [FLOW-FIELD] Раз на скільки кадрів перебудовувати маршрутні поля.
	// Джерел багато й вони рухаються, тож троттлинг замість «при зміні клітинки».
	flowRebuildEvery = 6

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
	pheromonesEnabled   = false // вмикає феромони фрустрації (стигмергію); false = чистий Q-learning без слідів
	epsilonDecayEnabled = false // ε: false = постійна (qEpsilonConst); true = автоспад max→min
	sharedBrain         = true  // true = всі учні ділять ОДНУ мережу (вулик-розум); false = кожен свою
	localSight          = true  // [POMDP] true = агент бачить гравця лише поблизу+по прямій; false = всевидющий
	aiPlayer            = false // [SELF-PLAY] true = гравцем керує мозок-жертва (вчиться тікати); false = людина
	useGRU              = true  // [RNN] true = рекурентна памʼять (GRU); false = frame-stacking (стек кадрів)
	friendlyFire        = false // [БІЙ] true = свої теж шкодять своїм (рій проріджує себе) — для експериментів
	// [FLOW-FIELD] Режим показу поля (клавіша V циклює):
	//   0 = вимкнено, 1 = поле ДО СТОРОНИ ГРАВЦЯ, 2 = поле ДО ВОРОГІВ
	showFlowField = 0
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
			X: playerSpawn.X,
			Y: playerSpawn.Y,
			// [БІЙ] HP > 0 → гравець витримує кілька ударів; MaxHP>0 ще й вмикає
			// малювання HP-бару в drawPixel (те саме, що у ворогів).
			HP:      playerMaxHP,
			MaxHP:   playerMaxHP,
			Faction: factionPlayer,
			Color:   color.RGBA{R: 0, G: 255, B: 180, A: 255},
			Label:   "Y}{IJIEC",
		},
		units: newUnits(),
	}

	// [SELF-PLAY] Даємо гравцю власний мозок-жертву (flee=true → reward за ВТЕЧУ).
	// Окрема ефемерна мережа: не зберігається (зберігаємо лише хижаків через saveBrains).
	if aiPlayer {
		game.player.Brain = NewBrain()
		game.player.Brain.flee = true
	}

	startBeat(1.0) // запускаємо аудіо перед стартом гри

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Boids")
	ebiten.SetTPS(120)

	if err := ebiten.RunGame(game); err != nil && err != errExit {
		log.Fatal(err)
	}
}
