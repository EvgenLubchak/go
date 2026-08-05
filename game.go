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
	units    []Pixel // [GO: SLICE] — динамічний масив, як Array в JS
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
	// Той самий патерн безпеки, що й boidMap: пишемо однопотоково (updateUnits),
	// читаємо паралельно (calcAcceleration) — фази не перетинаються, гонок нема.
	frustration [boidMapH][boidMapW]float32

	tick       int     // лічильник кадрів
	difficulty float32 // множник складності (1.0 = старт)

	// [ПАУЗА] Клавіша P. Стан ГРИ, а не глобальний прапорець: як gameOver.
	// НЕ плутати з frozenPolicy (клавіша L): та морозить НАВЧАННЯ, а світ живе;
	// пауза морозить світ, а навчання просто не отримує нових кадрів.
	paused bool

	attackCooldown int // кадрів до наступного удару
	attackTimer    int // кадрів до кінця анімації кола

	// [RAYCASTER] Вид від першої особи (Wolfenstein-стиль). Симуляція лишається
	// 2D — змінюється ЛИШЕ камера/рендер. Перемикач: клавіша F.
	firstPerson bool    // false = вид зверху; true = від першої особи
	camAngle    float32 // напрямок камери (рад), слідує за напрямком руху гравця

	metrics Metrics // [МЕТРИКИ] крива навчання рою (клавіша G)

	// [FLOW-FIELD] Два поля маршрутів крізь лабіринт (multi-source BFS):
	// одне веде до сторони гравця, друге — до ворогів. Кожна сторона читає те,
	// що веде до супротивника. Візуалізація — клавіша V (циклює поля).
	flowToPlayerSide FlowField
	flowToEnemySide  FlowField
	flowTick         int // лічильник перебудов (троттлинг)
}

// [GO: SENTINEL ERROR]
// В Go немає exceptions — функції повертають error як звичайне значення.
// errors.New() створює унікальну помилку для порівняння: err == errExit.
var errExit = errors.New("exit")

// restart скидає стан ГРИ до початкового — але НЕ мозки.
//
// Мережі (а з ними й буфери досвіду) переносяться в новий склад поля через
// newUnitsWithHive. Інакше кожен рестарт відкидав би все навчання від початку сесії:
// newUnits перечитує ваги з диска, а пишуться вони лише на виході й на game over.
// На навчальному стенді це коштувало б дорожче за саму зручність рестарту.
func (g *Game) restart() {
	// Збираємо живі мережі ДО того, як переберемо units.
	hive := map[string]*Net{}
	for i := range g.units {
		b := g.units[i].Brain
		if b != nil && b.net != nil && b.net.file != "" {
			hive[b.net.file] = b.net
		}
	}

	g.player.X = playerSpawn.X
	g.player.Y = playerSpawn.Y
	g.player.VelX = 0
	g.player.VelY = 0
	g.player.HP = playerMaxHP // [БІЙ] відновлюємо здоровʼя
	g.player.resetFur()
	g.player.InvulnTimer = 0
	g.attackCooldown = 0
	g.attackTimer = 0
	g.units = newUnitsWithHive(hive)

	// [МЕТРИКИ] Лічильники заміру описують СВІТ, а не навчання — тож на рестарті їх
	// треба скинути. Криві навчання при цьому лишаються: вони про мережу, яка
	// переживає рестарт.
	//
	// Без цього m.agents накопичував Brain-и через усі життя (голови нові, мережі ті
	// самі), і per-agent усереднювався по агентах із РІЗНИХ світів. На панелі це
	// читалось як «4 юніти на полі» при одному стражнику — саме так і зловили.
	g.metrics.resetCounters()

	g.gameOver = false
	g.paused = false // інакше рестарт із паузи давав би застиглий новий світ
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

	// P — [ПАУЗА] застиглий світ. Перемикачі виду нижче лишаються робочими: саме
	// вони й роблять паузу корисною — можна розглянути метрики, вуса, flow-field і
	// форму тіл у застиглому кадрі.
	if inpututil.IsKeyJustPressed(ebiten.KeyP) {
		g.paused = !g.paused
	}

	// B — вручну переключити музичний патерн (експерименти з ритмом). Раніше сиділа
	// на P; перевішана, бо при soundEnabled=false вона все одно нічого не робить,
	// а P потрібніша під паузу.
	if inpututil.IsKeyJustPressed(ebiten.KeyB) {
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

	// M — скинути лічильники заміру (blind-chase / catch). Тиснемо, коли рій уже
	// навчився → далі метрики відображають саме навчену політику, а не всю історію.
	if inpututil.IsKeyJustPressed(ebiten.KeyM) {
		g.metrics.resetCounters()
	}

	// L — [ЗАМІР] заморозити/розморозити політику: навчання off, ε=0.
	// Скидання лічильників вшите СЮДИ навмисно: у протоколі заморозка й скидання
	// завжди йдуть разом, а забути друге — найдешевший спосіб зіпсувати прогін
	// (вибірка тоді містила б хвіст ще-навчальних кадрів).
	if inpututil.IsKeyJustPressed(ebiten.KeyL) {
		frozenPolicy = !frozenPolicy
		if frozenPolicy {
			g.metrics.resetCounters()
		}
	}

	// R — [РЕСТАРТ] новий світ, ТІ САМІ мозки (див. restart). Працює завжди, а не
	// лише на game over: стенд переріс у гру, і перезапускати процес із консолі
	// заради нового забігу означало б щоразу губити накопичене навчання.
	if inpututil.IsKeyJustPressed(ebiten.KeyR) {
		g.restart()
	}

	// V — [FLOW-FIELD] циклює: вимк → поле до сторони гравця → поле до ворогів
	if inpututil.IsKeyJustPressed(ebiten.KeyV) {
		showFlowField = (showFlowField + 1) % 3
	}

	if g.gameOver {
		return nil // R обробляється вище — працює і тут, і в живій грі
	}

	// [ПАУЗА] Виходимо ДО g.tick++ — інакше час ішов би далі, і LVL (він же годинник
	// замірів) та вікна метрик пливли б, поки світ стоїть.
	if g.paused {
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
	g.updateFlowFields() // [FLOW-FIELD] маршрути обох сторін (троттлинг)
	g.playerAttack()
	g.updateBoidMap()
	g.calcAcceleration()
	g.trainBrains() // [SHARED BRAIN] навчання мереж хижаків раз/кадр, ОДНОПОТОКОВО
	if aiPlayer && !frozenPolicy && g.player.Brain != nil {
		g.player.Brain.net.train(qBatch) // [SELF-PLAY] тренуємо мозок-жертву
	}
	g.metrics.collect(g) // [МЕТРИКИ] збір показників навчання (однопотоково)
	g.updateUnits()
	g.resolveImpacts() // [БІЙ] шкода від удару на швидкості (після руху — швидкості свіжі)

	// [БІЙ] Розштовхуємо юнітів із гравця САМЕ ТУТ, після розрахунку шкоди.
	//
	// ПОРЯДОК КРИТИЧНИЙ. pushOffPlayer гасить складову швидкості, спрямовану в
	// гравця. Якби він відпрацьовував ДО resolveImpacts, то на кадрі прильоту удар
	// рахувався б по вже обнуленій швидкості — і жоден удар не зараховувався б
	// ніколи, рій став би повністю нешкідливим.
	//
	// А так виходить рівно задумане: юніт, який ЗАЙШОВ НА УДАР з дистанції, встигає
	// завдати шкоди своєю швидкістю прильоту, і лише потім його відштовхує. Поки він
	// притиснутий, швидкість гаситься щокадру, тож розігнатись до порогу (0.72) він
	// не встигає — за кадр мозок додає лише brainForce 0.3. Обійми більше не кусають.
	for i := range g.units {
		g.pushOffPlayer(&g.units[i])
	}
	g.handleDeadUnits()
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
	if frozenPolicy {
		return // [ЗАМІР] політика заморожена — ваги не рухаються
	}
	seen := map[*Net]bool{}
	for i := range g.units {
		b := g.units[i].Brain
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
	// [ДВА ВУЛИКИ] Зберігаємо КОЖНУ унікальну мережу у ЇЇ власний файл (Net.file):
	// рій і вбивці вчаться незалежно. Дедуплікація по вказівнику — той самий
	// патерн, що й у trainBrains. Ефемерні мережі (file == "") SaveNet пропустить.
	seen := map[*Net]bool{}
	for i := range g.units {
		b := g.units[i].Brain
		if b == nil || b.net == nil || seen[b.net] {
			continue
		}
		seen[b.net] = true
		SaveNet(b.net)
	}
}

// Layout — розмір логічного екрану (Ebiten масштабує під вікно).
func (g *Game) Layout(_, _ int) (int, int) {
	return screenWidth, screenHeight
}
