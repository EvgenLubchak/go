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
	IsLearner       bool       // true → створюємо Brain (Q-learning); незалежно від Label
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
	// Не використовує AggressionForce/PounceMulti — замість них Brain підбирає ваги сам.
	// MaxSpeed і DetectionRange задають фізичні межі, а рішення приймає нейрон.
	ConfigLearner = EnemyConfig{
		WanderStrength:  0.1, // мінімальне блукання для дослідження
		AlignmentRate:   0.0, // не флокується — думає сам
		CohesionRate:    0.0,
		SeparationRate:  0.01,
		MaxSpeed:        1.2, // середня швидкість
		AggressionForce: 0.0, // НЕ використовується — замість цього Brain
		BurstChance:     0.0,
		BurstForce:      0.0,
		DetectionRange:  300.0, // бачить далеко — щоб було що вивчати
		PounceMulti:     0.0,
		MaxHP:           2,                            // живучий — більше часу на навчання
		Color:           color.RGBA{0, 255, 100, 255}, // зелений — учень
		Label:           "",
		IsLearner:       true, // ← саме це вмикає мозок, а не мітка
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

// newEnemies створює slice з рівномірним розподілом усіх трьох типів.
//
// [GO: MODULO CYCLING]
// i % len(configs) циклічно перебирає типи: 0,1,2,0,1,2,...
func newEnemies(count int) []Pixel {
	//configs := []EnemyConfig{ConfigBoid, ConfigPredator, ConfigSpeeder, ConfigHP, ConfigGroup}
	configs := []EnemyConfig{ConfigLearner}
	enemies := make([]Pixel, count)

	// [SHARED BRAIN] У режимі sharedBrain усі учні ділять ОДНУ мережу (вулик-розум).
	// Створюємо її раз тут; нижче кожен Brain лише вказує на неї.
	var sharedNet *Net
	sharedLoaded := false
	if sharedBrain {
		if sharedNet = LoadNet(); sharedNet != nil {
			sharedLoaded = true
		} else {
			sharedNet = NewNet()
		}
	}

	for i := range enemies {
		cfg := configs[i%len(configs)]

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
			if sharedBrain {
				brain = NewBrainWith(sharedNet)
				if sharedLoaded {
					brain.age = qEpsilonDecay // завантажена = навчена → ε-floor
				}
			} else {
				net := LoadNet()
				loaded := net != nil
				if net == nil {
					net = NewNet()
				}
				brain = NewBrainWith(net)
				if loaded {
					brain.age = qEpsilonDecay // завантажений = навчений → ε-floor
				}
			}
		}

		enemies[i] = Pixel{
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
		}
	}
	return enemies
}
