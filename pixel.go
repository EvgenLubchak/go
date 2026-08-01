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

// EnemyConfig — параметри поведінки конкретного типу ворога.
// Замість глобальних констант — кожен тип несе свої налаштування.
//
// [GO: STRUCT AS CONFIG]
// Замість масиву констант — структура даних. Легко передавати, копіювати, розширювати.
type EnemyConfig struct {
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
	Count           int        // скільки таких виходить на поле (див. enemyRoster)
	IsLearner       bool       // true → створюємо Brain (Q-learning); незалежно від Label

	// [ВБИВЦЯ] true → мозок цього типу отримує на вхід FLOW-FIELD (напрямок до
	// цілі крізь стіни) замість прямого напрямку, і вчиться в ОКРЕМІЙ мережі
	// (свій вулик + свій файл ваг). Так тонко налаштований рій лишається цілим.
	UsesFlowField bool

	// [БІЙ] true → до нагороди додаються бойові члени (+ за завдану шкоду,
	// − за отриману, ++ за вбивство). Окремий прапорець від UsesFlowField, щоб
	// можна було вмикати їх незалежно (напр. дати бойову нагороду і рою — для A/B).
	CombatReward bool
}

// [GO: PACKAGE-LEVEL VAR]
// Три типи ворогів. Визначаються один раз, читаються скрізь у пакеті.
var (
	// ConfigBoid — стадний, помірний. Колір і агресія рандомні на старті.
	ConfigBoid = EnemyConfig{
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
	ConfigPredator = EnemyConfig{
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
	ConfigSpeeder = EnemyConfig{
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

	ConfigHP = EnemyConfig{
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
	ConfigLearner = EnemyConfig{
		WanderStrength:  0.1, // мінімальне блукання для дослідження
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
		Count:       20,                           // скільки їх на полі
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
	ConfigKiller = EnemyConfig{
		WanderStrength:  0.05, // майже без хаосу — він цілеспрямований
		AlignmentRate:   0.0,
		CohesionRate:    0.0,
		SeparationRate:  0.01,
		MaxSpeed:        1.6, // швидший за рій (1.2) → сильніший удар
		AggressionForce: 0.0, // не використовується — рішення приймає Brain
		BurstChance:     0.0,
		BurstForce:      0.0,
		DetectionRange:  0.0, // не впливає (лише debug-коло)
		PounceMulti:     0.0,
		Count:           5,                            // мало: вони сильніші за рій
		MaxHP:           3,                            // витримує на удар більше за рій
		Color:           color.RGBA{255, 90, 60, 255}, // червоний — щоб одразу вирізняти
		Label:           "",
		IsLearner:       true,
		UsesFlowField:   true, // ← окремий мозок + flow-field на вхід
		CombatReward:    true, // ← вчиться БИТИ, а не лише наздоганяти
	}

	ConfigGroup = EnemyConfig{
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
)

// enemyRoster — СКЛАД поля бою: які типи виходять і (через Count у кожному) по
// скільки. Прибрати тип = закоментувати рядок; додати новий = дописати рядок.
//
// Чому список, а не «перші N — вбивці»: раніше кількість задавалась двома
// незалежними числами (enemyCount + killerCount), і при enemyCount ≤ killerCount
// переслідувачі МОВЧКИ зникали. Тепер кожен тип відповідає сам за себе.
var enemyRoster = []EnemyConfig{
	ConfigLearner, // зелені: реактивний рій-переслідувач (baseline)
	ConfigKiller,  // червоні: знають лабіринт (flow-field) + бойова нагорода
}

// enemyTotal — скільки всього ворогів дає поточний склад.
func enemyTotal() int {
	n := 0
	for _, cfg := range enemyRoster {
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
	Color       color.RGBA
	Label       string
	Cfg         EnemyConfig // конфіг типу (порожній для гравця)
	Brain       *Brain      // нейронна мережа (nil для звичайних ворогів, не nil для Learner)
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

// newEnemies створює поле бою за enemyRoster: кожен тип дає cfg.Count юнітів.
//
// [ДВА ВУЛИКИ] Типи мозку вчаться НЕЗАЛЕЖНО: у кожного своя спільна мережа і
// свій файл ваг. Тому зміна reward/входів для вбивці не чіпає тонко налаштований
// рій — і навпаки.
func newEnemies() []Pixel {
	enemies := make([]Pixel, 0, enemyTotal())

	// [SHARED BRAIN] У режимі sharedBrain усі агенти ОДНОГО типу ділять одну
	// мережу (вулик-розум). Створюємо/вантажимо по одній на тип; нижче кожен
	// Brain лише вказує на потрібну.
	var chaserNet, killerNet *Net
	var chaserLoaded, killerLoaded bool
	if sharedBrain {
		chaserNet, chaserLoaded = newNetFor(brainFile)
		killerNet, killerLoaded = newNetFor(killerFile)
	}

	i := -1 // наскрізний номер юніта — для циклічного перебору спавн-точок
	for _, cfg := range enemyRoster {
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

			// Спавн: якщо в levelLayout є 'E' → циклічно по ним; інакше — рандом.
			// [GO: MODULO] i%len(enemySpawns) — циклічний перебір без виходу за межі.
			var spawnX, spawnY float32
			if len(enemySpawns) > 0 {
				sp := enemySpawns[i%len(enemySpawns)]
				spawnX, spawnY = sp.X, sp.Y
			} else {
				spawnX = float32(rand.Intn(screenWidth - pixelSize))
				spawnY = float32(rand.Intn(screenHeight - pixelSize))
			}

			// [GO: POINTER = nil для звичайних ворогів]
			// Brain створюємо тільки для Learner (cfg.IsLearner). Тригер — прапорець,
			// НЕ мітка. У режимі sharedBrain усі вказують на спільну мережу; інакше —
			// кожен має власну (завантажену з файлу або нову).
			var brain *Brain
			if cfg.IsLearner {
				// Файл ваг залежить від ТИПУ мозку — вбивця вчиться окремо від рою.
				file := brainFile
				net, loaded := chaserNet, chaserLoaded
				if cfg.UsesFlowField {
					file, net, loaded = killerFile, killerNet, killerLoaded
				}
				if !sharedBrain {
					net, loaded = newNetFor(file) // кожен агент — власна мережа
				}
				brain = NewBrainWith(net)
				brain.combat = cfg.CombatReward   // [БІЙ] бойові члени нагороди
				brain.flowNav = cfg.UsesFlowField // [ВБИВЦЯ] прогрес міряємо вздовж коридору
				if loaded {
					brain.age = qEpsilonDecay // завантажена = навчена → ε-floor
				}
			}

			enemies = append(enemies, Pixel{
				X:          spawnX,
				Y:          spawnY,
				VelX:       (rand.Float32() - 0.5) * cfg.MaxSpeed,
				VelY:       (rand.Float32() - 0.5) * cfg.MaxSpeed,
				Aggression: aggression,
				HP:         cfg.MaxHP,
				MaxHP:      cfg.MaxHP,
				Faction:    factionEnemy, // [БІЙ] увесь рій — одна фракція
				Color:      col,
				Label:      cfg.Label,
				Cfg:        cfg,
				Brain:      brain,
			})
		}
	}
	return enemies
}
