package main

import (
	"errors"
	"log"
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

	// [ЗАМОРОЗКА] Кадрів, які світ ще стоїть після удару, і кулдаун до наступної паузи.
	// Виставляє freezeOnHit (combat.go), читає ЛИШЕ Update — стенд про них не знає.
	hitstop     int
	hitstopCool int

	// [ПАУЗА] Клавіша P. Стан ГРИ, а не глобальний прапорець: як gameOver.
	// НЕ плутати з frozenPolicy (клавіша L): та морозить НАВЧАННЯ, а світ живе;
	// пауза морозить світ, а навчання просто не отримує нових кадрів.
	paused bool

	// [RAYCASTER] Вид від першої особи (Wolfenstein-стиль). Симуляція лишається
	// 2D — змінюється ЛИШЕ камера/рендер. Перемикач: клавіша F.
	firstPerson bool    // false = вид зверху; true = від першої особи
	camAngle    float32 // напрямок камери (рад), слідує за напрямком руху гравця

	// [ВУЛИКИ] Реєстр мереж: файл ваг → мережа. Живе на рівні ГРИ, а не в тілах.
	//
	// Причина конкретна й дорога. Раніше мережі були досяжні ЛИШЕ через g.units, а
	// handleDeadUnits видаляє юніта, у якого скінчились повернення. Тип, що вимер
	// повністю (у стражника Respawns: 1, тобто два життя), зникав зі зрізу разом зі
	// своєю мережею — і на виході його файл ваг не писався ВЗАГАЛІ. Уся сесія навчання
	// того типу губилась мовчки, без жодного повідомлення.
	//
	// Реєстр тримає мережу незалежно від того, чи лишилось живе тіло.
	hive map[string]*Net

	metrics Metrics // [МЕТРИКИ] крива навчання рою (клавіша M)

	// [FLOW-FIELD] Два поля маршрутів крізь лабіринт (multi-source BFS):
	// одне веде до сторони гравця, друге — до ворогів. Кожна сторона читає те,
	// що веде до супротивника. Візуалізація — панель Tab (шар flow-field).
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
	// Реєстр переживає рестарт сам собою — і саме тому типи, що вимерли ДО натискання
	// R, більше не губляться. Раніше тут стояв обхід g.units, і мережа зникала разом з
	// останнім тілом.
	if g.hive == nil {
		g.hive = map[string]*Net{}
	}
	// Але тіла можуть тримати мережі, яких у реєстрі ще НЕМАЄ: гра, зібрана в обхід
	// newUnitsWithHive (стенд, тести), заповнює лише Brain-и. Тож доповнюємо реєстр із
	// тіл, а не замінюємо його ними — інакше рестарт такої гри перечитав би ваги з
	// диска й викинув усе, що набралось за сесію. Цей регрес зловив уже наявний
	// TestRestartKeepsBrains, і саме так і мало статись.
	for i := range g.units {
		b := g.units[i].Brain
		if b == nil || b.net == nil || b.net.file == "" {
			continue
		}
		if _, ok := g.hive[b.net.file]; !ok {
			g.hive[b.net.file] = b.net
		}
	}

	g.player.X = playerSpawn.X
	g.player.Y = playerSpawn.Y
	g.player.VelX = 0
	g.player.VelY = 0
	g.player.HP = playerMaxHP // [БІЙ] відновлюємо здоровʼя
	g.player.resetFur()
	g.player.InvulnTimer = 0
	g.player.DashPhase, g.player.DashTimer = dashIdle, 0 // [РИВОК] обриваємо недоведену атаку
	// [SELF-PLAY] Ухилення жертви — так само: новий світ починається з чистого стану.
	g.player.DodgeTimer, g.player.DodgeCooldown, g.player.DodgeRecover = 0, 0, 0
	g.units = newUnitsWithHive(g.hive)

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
	// ESC: при відкритій панелі — ЗАКРИТИ ПАНЕЛЬ, інакше — вихід зі збереженням.
	// Звичка «ESC = закрити модалку» не має коштувати сесії. JustPressed (а не
	// IsKeyPressed, як було) навмисно: інакше той самий утримуваний ESC, що закрив
	// панель, наступного ж кадру вийшов би з гри.
	if inpututil.IsKeyJustPressed(ebiten.KeyEscape) {
		if panelOpen {
			panelOpen = false
		} else {
			g.saveBrains() // зберігаємо мозок перед виходом
			return errExit
		}
	}

	// Tab — [ПАНЕЛЬ НАЛАШТУВАНЬ] (див. panel.go). Гра під панеллю НЕ спиняється —
	// половина її ручок це прилади порівняння на живому кадрі; придушується лише
	// ввід гравця, бо стрілки/WASD віддані навігації.
	if inpututil.IsKeyJustPressed(ebiten.KeyTab) {
		panelOpen = !panelOpen
	}
	if panelOpen {
		handlePanelInput(g)
	}

	// P — [ПАУЗА] застиглий світ. Перемикачі виду нижче лишаються робочими: саме
	// вони й роблять паузу корисною — можна розглянути метрики, вуса, flow-field і
	// форму тіл у застиглому кадрі.
	if inpututil.IsKeyJustPressed(ebiten.KeyP) {
		g.paused = !g.paused
	}

	// Звук і барабанний патерн переїхали з клавіші B на панель (Tab): B була
	// мертвою при вимкненому звуці, а сам soundEnabled узагалі не мав ручки.

	// [ЗУМ] +/− наближають і віддаляють вид зверху. Камера тягнеться за гравцем із
	// відставанням і відсікається до меж світу — тож на 1.0 вона сама стає в центр
	// карти, і картинка тотожна тій, що була завжди. Окремого випадку не потрібно.
	if inpututil.IsKeyJustPressed(ebiten.KeyEqual) || inpututil.IsKeyJustPressed(ebiten.KeyKPAdd) {
		cam.zoom = clamp(cam.zoom+camZoomStep, camZoomMin, camZoomMax)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyMinus) || inpututil.IsKeyJustPressed(ebiten.KeyKPSubtract) {
		cam.zoom = clamp(cam.zoom-camZoomStep, camZoomMin, camZoomMax)
	}
	// Темп гри (TPS) переїхав із клавіші T на панель (Tab, «Ігрові налаштування») —
	// див. toggleTPS у panel.go. Персиститься в settings.json.

	// AA і SS переїхали з клавіш N/H на ПАНЕЛЬ (Tab, блок «Графіка») — перший крок
	// виносу перемикачів. Роль приладів порівняння не постраждала: гра під панеллю
	// живе, тож FPS видно тим самим оком. Зміни персистяться в settings.json
	// (лише відхилення від дефолтів — див. settings.go).

	// F — перемикач виду: зверху ↔ від першої особи (raycaster)
	if inpututil.IsKeyJustPressed(ebiten.KeyF) {
		g.firstPerson = !g.firstPerson
	}

	// M — перемикач панелі метрик (крива навчання). ⚠️ ОБМІНЯНА МІСЦЯМИ з G:
	// M мнемонічніша для «метрик», а скидання переїхало на G. Стара звичка
	// «G = показати метрики» тепер СКИДАЄ лічильники — перевчитись свідомо.
	if inpututil.IsKeyJustPressed(ebiten.KeyM) {
		showMetrics = !showMetrics
	}

	// G — скинути лічильники заміру (blind-chase / catch). Тиснемо, коли рій уже
	// навчився → далі метрики відображають саме навчену політику, а не всю історію.
	// (До обміну жила на M — див. коментар вище.)
	if inpututil.IsKeyJustPressed(ebiten.KeyG) {
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

	// 2 — [DOUBLE DQN] перемикач цілі Беллмана наживо: класична (max по target) ↔
	// подвійна (argmax живої, оцінка target). Наживо, бо це і є експеримент:
	// Q-барометр стражника на панелі при обох цілях, прогнози — у roadmap.
	// Індикатор ddqn — у рядку конфігурації панелі, скріни самодокументуються.
	if inpututil.IsKeyJustPressed(ebiten.KeyDigit2) {
		doubleDQN = !doubleDQN
	}

	// R — [РЕСТАРТ] новий світ, ТІ САМІ мозки (див. restart). Працює завжди, а не
	// лише на game over: стенд переріс у гру, і перезапускати процес із консолі
	// заради нового забігу означало б щоразу губити накопичене навчання.
	if inpututil.IsKeyJustPressed(ebiten.KeyR) {
		g.restart()
	}

	// Шар flow-field переїхав із клавіші V на панель (Tab): тристановий цикл
	// «вимк → до сторони гравця → до ворогів», тепер в обидва боки.

	if g.gameOver {
		return nil // R обробляється вище — працює і тут, і в живій грі
	}

	// [ЗАМОРОЗКА] Світ стоїть кілька кадрів після удару за участю гравця (див.
	// hitstopDealt у tuning_combat.go). Місце вибране з трьох міркувань:
	//
	//	ПІСЛЯ ВСІХ КЛАВІШ — морозимо СВІТ, а не ВВЕДЕННЯ. Якби вийшли раніше,
	//	   натискання, зроблене й відпущене за ці кадри, зникло б безслідно:
	//	   IsKeyJustPressed його б не побачив. Навіть 50 мс — досяжна ціль для пальця,
	//	   а загублений ривок читався б як «гра не слухається»;
	//	ДО cam.follow — інакше камера продовжувала б їхати, і завмирання світу поруч
	//	   із рухомим кадром читалось би як підвисання, а не як удар;
	//	ОКРЕМО ВІД ПАУЗИ — пауза лишає cam.follow робочим навмисно: саме він відсікає
	//	   центр огляду до меж світу, і без нього зум на паузі показав би порожнечу
	//	   за краєм карти.
	if g.hitstop > 0 {
		g.hitstop--
		return nil
	}
	if g.hitstopCool > 0 {
		g.hitstopCool--
	}

	cam.follow(g.player.X+pixelSize/2, g.player.Y+pixelSize/2)

	// [ПАУЗА] Виходимо ДО g.tick++ — інакше час ішов би далі, і LVL (він же годинник
	// замірів) та вікна метрик пливли б, поки світ стоїть.
	if g.paused {
		return nil
	}

	// % — залишок від ділення (як в PHP/JS). tick%levelUpEvery==0 → новий рівень.
	g.tick++
	if difficultyGrowth && g.tick%levelUpEvery == 0 && g.difficulty < maxDifficulty {
		g.difficulty += difficultyStep
		advanceBeat(g.difficulty) // темп зростає щорівня, патерн чергується
	}

	if aiPlayer && g.player.Brain != nil {
		g.updatePrey() // [SELF-PLAY] гравцем керує мозок-жертва
	} else if !panelOpen { // [ПАНЕЛЬ] стрілки/WASD/Space віддані навігації панелі
		// [РИВОК] Пробіл читаємо ПЕРЕД рухом: напрямок удару беремо з клавіш, а
		// handlePlayerInput у фазах атаки все одно нічого не додасть.
		g.playerDashInput()
		g.handlePlayerInput()
	}
	g.updatePlayer()
	g.updateFlowFields() // [FLOW-FIELD] маршрути обох сторін (троттлинг)
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
	save := func(n *Net) {
		if n == nil || seen[n] {
			return
		}
		seen[n] = true
		// Помилку НЕ ковтаємо: невдале збереження — це мовчки втрачена сесія
		// навчання, найгірший клас вади в цьому проєкті. Гру не валимо (вихід і
		// так триває), але слід у консолі лишається.
		if err := SaveNet(n); err != nil {
			log.Printf("saveBrains: %s: %v", n.file, err)
		}
	}
	// [ВИМЕРЛІ ТИПИ] Спершу реєстр: там мережа лишається й тоді, коли жодного тіла
	// цього типу вже немає. Саме цього бракувало — тип із скінченними поверненнями
	// вимирав, і його файл ваг не писався взагалі.
	for _, n := range g.hive {
		save(n)
	}
	// Потім живі тіла: у режимі sharedBrain=false реєстр порожній за побудовою
	// (кожен агент має власну мережу), і без цього проходу ми б не зберегли нічого.
	for i := range g.units {
		if b := g.units[i].Brain; b != nil {
			save(b.net)
		}
	}
}

// Layout — розмір логічного екрану (Ebiten масштабує під вікно).
func (g *Game) Layout(_, _ int) (int, int) {
	return screenWidth, screenHeight
}
