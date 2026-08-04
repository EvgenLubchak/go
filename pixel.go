package main

import (
	"image/color"
	"math/rand"
)

// [БІЙ] Фракції — «хто кому свій». Поки дві: гравець (+ у майбутньому його юніти)
// і рій. Далі на цьому виростуть командні бої: вбивці vs дружні боти.
const (
	factionPlayer = 0 // гравець і його союзники
	factionEnemy  = 1 // рій-хижак
)

// UnitConfig — параметри поведінки конкретного типу ворога.
// Замість глобальних констант — кожен тип несе свої налаштування.
//
// [GO: STRUCT AS CONFIG]
// Замість масиву констант — структура даних. Легко передавати, копіювати, розширювати.
type UnitConfig struct {
	WanderStrength  float32    // сила випадкового блукання
	AlignmentRate   float32    // сила вирівнювання до зграї (boids)
	CohesionRate    float32    // сила притягування до центру маси сусідів
	SeparationRate  float32    // сила відштовхування від кожного сусіда (ближче = сильніше)
	MaxSpeed        float32    // стеля швидкості
	AggressionForce float32    // базова сила переслідування гравця
	BurstChance     float32    // ймовірність поштовху за кадр
	BurstForce      float32    // сила поштовху (додається до швидкості)
	DetectionRange  float32    // радіус "відчуття" гравця (px)
	PounceMulti     float32    // множник кидка при зближенні
	MaxHP           int        // початкове HP
	Color           color.RGBA // базовий колір; A==0 → колір визначається Aggression
	Label           string     // мітка всередині пікселя (ЛИШЕ відображення)
	Count           int        // скільки таких виходить на поле (див. unitRoster)
	Faction         int        // [КОМАНДИ] чий юніт: factionEnemy (рій) чи factionPlayer (твої)
	WeightsFile     string     // файл ваг ЦЬОГО типу мозку — різні типи вчаться незалежно
	IsLearner       bool       // true → створюємо Brain (Q-learning); незалежно від Label

	// [ВБИВЦЯ] true → мозок цього типу отримує на вхід FLOW-FIELD (напрямок до
	// цілі крізь стіни) замість прямого напрямку, і вчиться в ОКРЕМІЙ мережі
	// (свій вулик + свій файл ваг). Так тонко налаштований рій лишається цілим.
	UsesFlowField bool

	// [БІЙ] true → до нагороди додаються бойові члени (+ за завдану шкоду,
	// − за отриману, ++ за вбивство). Окремий прапорець від UsesFlowField, щоб
	// можна було вмикати їх незалежно (напр. дати бойову нагороду і рою — для A/B).
	CombatReward bool

	// [ПАМʼЯТЬ] Контракт памʼяті ЦЬОГО типу. Нуль/MemoryDefault = взяти глобаль
	// (useGRU / memFrames / stackSkip у main.go і brain.go).
	//
	// Тристан у Memory навмисно: із простим bool неможливо відрізнити «явно стек» від
	// «не задано», і глобаль перестала б працювати як дефолт.
	//
	// Контракт стає властивістю МЕРЕЖІ, тож два типи з різною памʼяттю обовʼязково
	// мусять мати різні WeightsFile — інакше вони поділили б одну мережу з одним
	// контрактом, і другий тип тихо отримав би чужу форму входу.
	Memory    MemoryKind
	MemFrames int // 0 = глобальний memFrames
	StackSkip int // 0 = глобальний stackSkip
}

// [GO: PACKAGE-LEVEL VAR]
// Три типи ворогів. Визначаються один раз, читаються скрізь у пакеті.
var (
	// ConfigBoid — стадний, помірний. Колір і агресія рандомні на старті.
	ConfigBoid = UnitConfig{
		WanderStrength:  0.2,
		AlignmentRate:   0.03,
		CohesionRate:    0.002,
		SeparationRate:  0.03,
		MaxSpeed:        0.9,
		AggressionForce: 0.04,
		BurstChance:     0.0001,
		BurstForce:      50.0,
		DetectionRange:  80.0,
		PounceMulti:     7.0,
		MaxHP:           2,
		// Color.A == 0 → aggressionColor() визначить колір
		Label: "",
	}

	// ConfigPredator — повільний але смертоносний: великий радіус, сильний кидок.
	ConfigPredator = UnitConfig{
		WanderStrength:  0.1,
		AlignmentRate:   0.01,
		CohesionRate:    0.0005,
		SeparationRate:  0.01,
		MaxSpeed:        1.4,
		AggressionForce: 0.12,
		BurstChance:     0.0003,
		BurstForce:      80.0,
		DetectionRange:  180.0,
		PounceMulti:     14.0,
		MaxHP:           5,
		Color:           color.RGBA{220, 50, 50, 255},
		Label:           "",
	}

	// ConfigSpeeder — хаотичний, дуже швидкий, крихкий, майже не флокується.
	ConfigSpeeder = UnitConfig{
		WanderStrength:  0.6,
		AlignmentRate:   0.005,
		CohesionRate:    0.0001,
		SeparationRate:  0.05,
		MaxSpeed:        1.6,
		AggressionForce: 0.02,
		BurstChance:     0.004,
		BurstForce:      120.0,
		DetectionRange:  40.0,
		PounceMulti:     2.0,
		MaxHP:           1,
		Color:           color.RGBA{200, 220, 50, 255},
		Label:           "",
	}

	ConfigHP = UnitConfig{
		WanderStrength:  0.01,
		AlignmentRate:   0.0001,
		CohesionRate:    0.00001,
		SeparationRate:  0.1,
		MaxSpeed:        0.3,
		AggressionForce: 0,
		BurstChance:     0.01,
		BurstForce:      10,
		DetectionRange:  400,
		PounceMulti:     1,
		MaxHP:           1,
		Color:           color.RGBA{255, 255, 255, 255},
		Label:           "",
	}

	// ConfigLearner — ворог-учень з нейронною мережею замість захардкоджених правил.
	// Не використовує AggressionForce/PounceMulti/DetectionRange — замість них Brain
	// підбирає ваги сам. Фізичну межу задає лише MaxSpeed; рішення приймає нейрон.
	//
	// [RL] ЧОМУ В УЧНІВ НЕМАЄ БЛУКАННЯ (WanderStrength = 0)
	//
	// Блукання додається ПРЯМО У ШВИДКІСТЬ (updateUnits), а тертя в нас немає
	// (damping = 1) — отже випадкові штовхачі не гаснуть, а накопичуються у
	// випадкове блукання по швидкості, обмежене лише стелею MaxSpeed. При 0.1 за
	// кадр це набігало до ~40% максимальної швидкості.
	//
	// Але головне не величина, а те, що воно ЛАМАЄ CREDIT ASSIGNMENT. Нагорода в
	// нас — швидкість зближення, тобто ВЛАСНИЙ рух агента. Частину цього руху агент
	// не обирав: його штовхнуло блукання. Він отримував плюс за зближення, якого не
	// робив, і мінус за віддалення, якого не хотів. Це той самий клас помилки, що
	// колись була з рухом ГРАВЦЯ в нагороді, і той самий контрольний тест:
	// «чи міг агент вплинути на це число сам?» — блукання його не проходить.
	//
	// Дослідження при цьому нікуди не зникло: ε-greedy робить випадковою ДІЮ, за
	// яку агент потім чесно відповідає. Блукання ж підмішувало рух повз усю систему
	// навчання — другий механізм тієї самої функції, тільки гірший.
	//
	// SeparationRate лишається: це фізика (юніти не мають злипатись у точку), і без
	// неї удар на швидкості зближення втрачає сенс. Він теж невидимий для агента —
	// у входах немає жодної інформації про сусідів, — але його внесок на порядок
	// менший, і без нього ламається бій.
	ConfigLearner = UnitConfig{
		WanderStrength:  0.0, // [RL] 0 — див. «Чому в учнів немає блукання» нижче
		AlignmentRate:   0.0, // не флокується — думає сам
		CohesionRate:    0.0,
		SeparationRate:  0.01,
		MaxSpeed:        1.2, // середня швидкість
		AggressionForce: 0.0, // НЕ використовується — замість цього Brain
		BurstChance:     0.0,
		BurstForce:      0.0,
		DetectionRange:  300.0, // НЕ впливає на учня (лише debug-коло showDetectionCircle);
		//                        зір мозку — це sightRange (POMDP) + whiskerRange (вуса)
		PounceMulti: 0.0,
		Count:       15, // скільки їх на полі
		Faction:     factionEnemy,
		WeightsFile: brainFile,
		MaxHP:       2,                            // живучий — більше часу на навчання
		Color:       color.RGBA{0, 255, 100, 255}, // зелений — учень
		Label:       "",
		IsLearner:   true, // ← саме це вмикає мозок, а не мітка
	}

	// ConfigKiller — [ВБИВЦЯ] мисливець, що ЗНАЄ ЛАБІРИНТ. На відміну від рою
	// (реактивний: тисне в бік, де «відчуває» ціль, і тому б'ється об стіни), він
	// отримує напрямок flow-field як вхід → ходить коридорами. Вчиться в окремій
	// мережі (killerFile), тож рій лишається недоторканим.
	//
	// Швидший і живучіший за рій — щоб «кидок кобри» був відчутним, але їх мало
	// (Count), інакше бій перетвориться на бійню.
	ConfigKiller = UnitConfig{
		WanderStrength:  0.0, // [RL] 0 — блукання ламало credit assignment
		AlignmentRate:   0.0,
		CohesionRate:    0.0,
		SeparationRate:  0.01,
		MaxSpeed:        1.6, // швидший за рій (1.2) → сильніший удар
		AggressionForce: 0.0, // не використовується — рішення приймає Brain
		BurstChance:     0.0,
		BurstForce:      0.0,
		DetectionRange:  0.0, // не впливає (лише debug-коло)
		PounceMulti:     0.0,
		Count:           3, // мало: вони сильніші за рій
		Faction:         factionEnemy,
		WeightsFile:     killerFile,
		MaxHP:           4,                            // витримує на удар більше за рій
		Color:           color.RGBA{255, 90, 60, 255}, // червоний — щоб одразу вирізняти
		Label:           "",
		IsLearner:       true,
		UsesFlowField:   true, // ← окремий мозок + flow-field на вхід
		CombatReward:    true, // ← вчиться БИТИ, а не лише наздоганяти
	}

	// ConfigAllyChaser — [КОМАНДИ] ЮНІТ ГРАВЦЯ. Полює не на гравця, а на найближчого
	// ВОРОГА (ціль обирається за фракцією, див. nearestHostile). Перехоплює рій,
	// поки той іде по тебе.
	//
	// Реактивний аналог рою, але на твоєму боці: свій вулик, бойова нагорода.
	ConfigAllyChaser = UnitConfig{
		WanderStrength:  0.0, // [RL] як у рою — жодного некерованого руху
		AlignmentRate:   0.0, // не флокується — думає сам
		CohesionRate:    0.0,
		SeparationRate:  0.01,
		MaxSpeed:        1.4, // трохи швидший за рій (1.2), повільніший за вбивцю (1.6)
		AggressionForce: 0.0, // не використовується — рішення приймає Brain
		BurstChance:     0.0,
		BurstForce:      0.0,
		DetectionRange:  0.0,
		PounceMulti:     0.0,
		Count:           20,
		Faction:         factionPlayer, // ← свій; рій його атакує, він рій
		WeightsFile:     allyFile,
		MaxHP:           3,
		Color:           color.RGBA{80, 170, 255, 255}, // блакитний — свої
		Label:           "",
		IsLearner:       true,
		CombatReward:    true, // бійці: + за шкоду, − за отриману, ++ за вбивство
	}

	// ConfigAllyKiller — [КОМАНДИ] ТВІЙ ВБИВЦЯ: та сама роль, що й ворожий, але на
	// твоєму боці. Читає ІНШЕ поле (flowToEnemySide) — маршрут до найближчого ворога
	// крізь лабіринт, тож працює симетрично до червоних.
	//
	// Вулик окремий: ворожий вбивця ганяє гравця, що тікає, твій — б'ється з ШІ.
	// Хочеш перевірити, чи роль ПЕРЕНОСИТЬСЯ між сторонами — постав обом однаковий
	// WeightsFile: мережа бачить лише ВІДНОСНІ входи («напрямок до моєї цілі»), тож
	// політика має бути та сама, зате досвіду вдвічі більше.
	ConfigAllyKiller = UnitConfig{
		WanderStrength:  0.0, // [RL] як у вбивці
		AlignmentRate:   0.0,
		CohesionRate:    0.0,
		SeparationRate:  0.01,
		MaxSpeed:        1.6, // як у ворожого вбивці — сторони симетричні
		AggressionForce: 0.0,
		BurstChance:     0.0,
		BurstForce:      0.0,
		DetectionRange:  0.0,
		PounceMulti:     0.0,
		Count:           2,
		Faction:         factionPlayer,
		WeightsFile:     allyKillerFile,
		MaxHP:           5,
		Color:           color.RGBA{140, 100, 255, 255}, // фіолетовий — твій вбивця
		Label:           "≡_≡",
		IsLearner:       true,
		UsesFlowField:   true, // ← поле до ВОРОГІВ (flowFor обирає за фракцією)
		CombatReward:    true,
	}

	ConfigGroup = UnitConfig{
		WanderStrength:  0.02,  // майже без хаосу — плавний рух
		AlignmentRate:   0.08,  // сильно рівняється на сусідів (головний пріоритет)
		CohesionRate:    0.005, // сильно тягнеться до центру групи
		SeparationRate:  0.02,  // не накладається, але тримається близько
		MaxSpeed:        0.7,   // повільний — не розбігається
		AggressionForce: 0.01,  // майже ігнорує гравця
		BurstChance:     0.0,   // ніяких ривків
		BurstForce:      0.0,
		DetectionRange:  50.0, // маленький радіус — реагує тільки впритул
		PounceMulti:     1.0,
		MaxHP:           5,
		Color:           color.RGBA{100, 180, 255, 255}, // блакитний — виділяється
		Label:           "",
	}

	// ConfigBoss — [БОС] ОДИНАК із власною мережею. Єдиний тип, для якого памʼять за
	// нашими замірами взагалі щось важить.
	//
	// ЧОМУ ОДИНАК. Виміряно, що вся перевага рою — це СПІЛЬНИЙ БУФЕР ДОСВІДУ, а не
	// кількість тіл: вісім юнітів з окремими мережами не кращі за одне (p = 0.14), а
	// спільний проти незалежних дає домінування 85% (p = 4×10⁻⁵). Бос стоїть саме на
	// гіршій стороні цього виміру — і саме тому памʼять для нього не декорація, а
	// єдиний канал інформації про зниклу ціль (+2.5 пункти, p = 0.001).
	//
	// UsesFlowField: false — КРИТИЧНО. У вбивці поле є, і саме тому йому памʼять не
	// потрібна: він ЗАВЖДИ знає шлях до цілі, згадувати нічого. Дати босу і поле, і
	// памʼять означало б зробити ще одного вбивцю з декоративним GRU. Ціна: бос
	// натикатиметься на стіни, як рій. Це свідомий обмін.
	//
	// MaxHP високий не лише для «боса»: власна мережа з одним тілом набирає досвід у
	// вісім разів повільніше за вулик, тож бос МУСИТЬ виживати, щоб узагалі вчитись.
	// Живучість тут — умова навчання, а не тільки баланс.
	//
	// Ваги в окремому файлі: рій ти скидаєш регулярно, а бос накопичується через усі
	// сесії й ніколи не забуває. Він вчиться повільно — але назавжди.
	//
	// ПАМʼЯТЬ ЗАДАНА ЯВНО (Memory: MemoryStack), а не успадкована з глобалі.
	//
	// Це рішення, а не дефолт: GRU на замірах виявився нерозрізненним від ВІДСУТНОСТІ
	// памʼяті навіть на одинаку (52% і 44% домінування при двох темпах навчання,
	// p = 0.76 і 0.37), тож бос на GRU вийшов би гіршим за звичайного юніта рою.
	// Стек — єдина конфігурація памʼяті, доведена на одинаку (65%, p = 0.004).
	//
	// Явно ще й тому, що тепер глобальний useGRU — лише ДЕФОЛТ: перемикаючи його для
	// експериментів, ти більше не зачепиш боса випадково. Коли важіль BPTT буде
	// полагоджено (див. коментар до useGRU), тут зміниться одне слово на MemoryGRU.
	ConfigBoss = UnitConfig{
		WanderStrength:  0.0, // жодного некерованого руху (див. ConfigLearner)
		AlignmentRate:   0.0, // одинак — з ким йому флокуватись
		CohesionRate:    0.0,
		SeparationRate:  0.01, // фізика: не злипатись із роєм
		MaxSpeed:        1.3,  // швидший за рій (1.2), повільніший за вбивцю (1.6)
		AggressionForce: 0.0,  // не використовується — рішення приймає Brain
		BurstChance:     0.0,
		BurstForce:      0.0,
		DetectionRange:  0.0,
		PounceMulti:     0.0,
		Count:           1, // ОДИН. У цьому вся суть типу
		Faction:         factionEnemy,
		WeightsFile:     bossFile,                      // ← власний вулик виникає САМ (мапа по файлу)
		MaxHP:           45,                            // умова навчання, не лише баланс
		Color:           color.RGBA{235, 70, 160, 255}, // малиновий — не сплутати ні з ким
		Label:           "$_$",
		IsLearner:       true,
		UsesFlowField:   false,       // ← памʼять має на що працювати лише без поля
		CombatReward:    true,        // бос мусить вчитись БИТИ, а не наздоганяти
		Memory:          MemoryStack, // ← ЯВНО, не з глобалі (див. вище)
	}
)

// unitRoster — СКЛАД поля бою: які типи виходять і (через Count у кожному) по
// скільки. Прибрати тип = закоментувати рядок; додати новий = дописати рядок.
//
// Чому список, а не «перші N — вбивці»: раніше кількість задавалась двома
// незалежними числами (enemyCount + killerCount), і при enemyCount ≤ killerCount
// переслідувачі МОВЧКИ зникали. Тепер кожен тип відповідає сам за себе.
var unitRoster = []UnitConfig{
	ConfigLearner,    // зелені: ворожий рій-переслідувач (baseline)
	ConfigKiller,     // червоні: вороги, що знають лабіринт (flow-field) + бойова нагорода
	ConfigAllyChaser, // блакитні: твої переслідувачі — перехоплюють рій
	ConfigAllyKiller, // фіолетові: твої вбивці — знають лабіринт, ідуть на ворога
	ConfigBoss,       // малиновий: ОДИНАК із власною мережею — єдиний, кому памʼять потрібна
}

// unitTotal — скільки всього ворогів дає поточний склад.
func unitTotal() int {
	n := 0
	for _, cfg := range unitRoster {
		n += cfg.Count
	}
	return n
}

// Pixel описує одну частинку — гравця або ворога.
type Pixel struct {
	X, Y       float32
	VelX, VelY float32 // velocity — вектор швидкості (куди і як швидко рухається)
	AccX, AccY float32 // acceleration — вектор прискорення (сума сил: alignment + chase)
	Aggression float32 // 0.0..1.0: для Boid — рандомний, для інших — 1.0
	HP, MaxHP  int
	HitTimer   int
	HitWall    bool // [RL] чи врізався у стіну цього кадру (сигнал штрафу для Brain)

	// [БІЙ] Кому належить юніт (свої не б'ють своїх, якщо friendlyFire=false).
	Faction int
	// [БІЙ] Кадри невразливості після отриманого удару. Без цього агенти, що
	// перекриваються, отримували б шкоду КОЖЕН кадр (миттєва смерть у купі).
	InvulnTimer int

	// [БІЙ] Кадри «відльоту» після зарахованого удару: поки > 0, стеля швидкості
	// піднята (knockSpeedMulti), інакше кліп MaxSpeed зʼїв би віддачу за один кадр.
	KnockTimer int
	// [ВОРС] Позиції суглобів кожної ворсинки у СВІТОВИХ координатах:
	// Fur[ворсинка][суглоб][x,y], суглоб 0 = середина, 1 = кінчик.
	// Світові (а не локальні) саме тому, що відставання має бути від РУХУ юніта.
	Fur [furStrands][2][2]float32

	// [ТІЛО] Стан пружини розміру: поточний масштаб і його швидкість. Пульс медузи
	// виникає з перельоту цієї пружини, а не з намальованого циклу.
	BodyScale float32
	BodyVel   float32

	Color color.RGBA
	Label string
	Cfg   UnitConfig // конфіг типу (порожній для гравця)
	Brain *Brain     // нейронна мережа (nil для звичайних ворогів, не nil для Learner)
}

// resetFur миттєво ставить ворс у спокійне положення навколо юніта.
// Потрібно при спавні/респавні/рестарті — інакше ворс «прилетів» би через пів
// карти з попередньої позиції (лаг чесно відпрацював би телепорт).
func (p *Pixel) resetFur() {
	p.BodyScale, p.BodyVel = 1, 0

	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2
	for i := 0; i < furStrands; i++ {
		dx, dy := dirs8[i][0], dirs8[i][1]
		p.Fur[i][0][0], p.Fur[i][0][1] = cx+dx*furLen*0.5, cy+dy*furLen*0.5
		p.Fur[i][1][0], p.Fur[i][1][1] = cx+dx*furLen, cy+dy*furLen
	}
}

// aggressionColor повертає колір від синього (пасивний) до червоного (агресивний).
func aggressionColor(a float32) color.RGBA {
	return color.RGBA{
		R: uint8(80 + 175*a),
		G: uint8(80 * (1 - a)),
		B: uint8(200 * (1 - a)),
		A: 255,
	}
}

// newUnits створює поле бою за unitRoster: кожен тип дає cfg.Count юнітів.
//
// [ДВА ВУЛИКИ] Типи мозку вчаться НЕЗАЛЕЖНО: у кожного своя спільна мережа і
// свій файл ваг. Тому зміна reward/входів для вбивці не чіпає тонко налаштований
// рій — і навпаки.
func newUnits() []Pixel { return newUnitsWithHive(nil) }

// newUnitsWithHive — те саме, але з можливістю ПЕРЕДАТИ вже наявні мережі.
//
// Навіщо: рестарт (клавіша R) мусить скидати СВІТ, а не мозки. Без цього кожне
// натискання R відкидало б усе навчання від початку сесії, бо newNetFor перечитує
// ваги з диска, а пишуться вони лише на виході й на game over. Для боса це було б
// смертельно: він і так вчиться у вісім разів повільніше за вулик, бо в нього немає
// спільного буфера досвіду.
//
// Передана мапа зберігає не лише ваги, а й БУФЕР ДОСВІДУ (він живе в Net) — а він
// набирається тисячі кадрів, і губити його на кожен рестарт означало б обнуляти
// найдорожче.
//
// nil = звичайний старт: мережі беруться з файлів.
// У режимі sharedBrain=false мапа не задіяна — там мережа в кожного своя, а тіла
// після рестарту нові.
func newUnitsWithHive(hive map[string]*Net) []Pixel {
	units := make([]Pixel, 0, unitTotal())

	// [ВУЛИКИ] У режимі sharedBrain агенти одного ТИПУ мозку (= одного файлу ваг)
	// ділять одну мережу. Мапа замість окремих змінних — щоб додати новий тип було
	// достатньо вказати йому WeightsFile, не чіпаючи цей код.
	if hive == nil {
		hive = map[string]*Net{}
	}
	hiveLoaded := map[string]bool{}
	for file := range hive {
		hiveLoaded[file] = true // передана мережа = вже навчена → ε-floor
	}
	netFor := func(file string, mem memContract) (*Net, bool) {
		if !sharedBrain {
			return newNetFor(file, mem) // кожен агент — власна мережа
		}
		if n, ok := hive[file]; ok {
			return n, hiveLoaded[file]
		}
		n, loaded := newNetFor(file, mem)
		hive[file], hiveLoaded[file] = n, loaded
		return n, loaded
	}

	i := -1 // наскрізний номер юніта — для циклічного перебору спавн-точок
	for _, cfg := range unitRoster {
		for k := 0; k < cfg.Count; k++ {
			i++

			// Boid: колір і агресія рандомні. Інші: фіксований колір, агресія 1.0.
			// [GO: ZERO VALUE CHECK] cfg.Color.A == 0 → Color не виставлений → Boid
			aggression := float32(1.0)
			col := cfg.Color
			if col.A == 0 {
				aggression = rand.Float32()
				col = aggressionColor(aggression)
			}

			// Спавн — за ФРАКЦІЄЮ: свої з міток 'A', вороги з 'E' (level.go).
			// Якщо міток немає — випадкова відкрита клітинка.
			// [GO: MODULO] i%len(spawns) — циклічний перебір без виходу за межі.
			spawns := enemySpawns
			if cfg.Faction == factionPlayer {
				spawns = allySpawns
			}
			var spawnX, spawnY float32
			if len(spawns) > 0 {
				sp := spawns[i%len(spawns)]
				spawnX, spawnY = sp.X, sp.Y
			} else {
				for tries := 0; tries < 50; tries++ {
					spawnX = float32(rand.Intn(screenWidth - pixelSize))
					spawnY = float32(rand.Intn(screenHeight - pixelSize))
					if !isInteriorWallRect(spawnX, spawnY) {
						break
					}
				}
			}

			// [GO: POINTER = nil для звичайних ворогів]
			// Brain створюємо тільки для Learner (cfg.IsLearner). Тригер — прапорець,
			// НЕ мітка. У режимі sharedBrain усі вказують на спільну мережу; інакше —
			// кожен має власну (завантажену з файлу або нову).
			var brain *Brain
			if cfg.IsLearner {
				mem := resolveMemContract(cfg.Memory, cfg.MemFrames, cfg.StackSkip)
				net, loaded := netFor(cfg.WeightsFile, mem)
				brain = NewBrainWith(net)
				brain.combat = cfg.CombatReward   // [БІЙ] бойові члени нагороди
				brain.flowNav = cfg.UsesFlowField // [ВБИВЦЯ] прогрес міряємо вздовж коридору
				if loaded {
					brain.age = qEpsilonDecay // завантажена = навчена → ε-floor
				}
			}

			units = append(units, Pixel{
				X:          spawnX,
				Y:          spawnY,
				VelX:       (rand.Float32() - 0.5) * cfg.MaxSpeed,
				VelY:       (rand.Float32() - 0.5) * cfg.MaxSpeed,
				Aggression: aggression,
				HP:         cfg.MaxHP,
				MaxHP:      cfg.MaxHP,
				Faction:    cfg.Faction, // [КОМАНДИ] сторона юніта
				Color:      col,
				Label:      cfg.Label,
				Cfg:        cfg,
				Brain:      brain,
			})
			units[len(units)-1].resetFur()
		}
	}
	return units
}
