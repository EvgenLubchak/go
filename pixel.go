package main

import (
	"image/color"
	"math/rand"
)

// EnemyConfig — параметри поведінки конкретного типу ворога.
// Замість глобальних констант — кожен тип несе свої налаштування.
//
// [GO: STRUCT AS CONFIG]
// Замість масиву констант — структура даних. Легко передавати, копіювати, розширювати.
type EnemyConfig struct {
	WanderStrength  float32    // сила випадкового блукання
	AlignmentRate   float32    // сила вирівнювання до зграї (boids)
	MaxSpeed        float32    // стеля швидкості
	AggressionForce float32    // базова сила переслідування гравця
	BurstChance     float32    // ймовірність поштовху за кадр
	BurstForce      float32    // сила поштовху (додається до швидкості)
	DetectionRange  float32    // радіус "відчуття" гравця (px)
	PounceMulti     float32    // множник кидка при зближенні
	MaxHP           int        // початкове HP
	Color           color.RGBA // базовий колір; A==0 → колір визначається Aggression
	Label           string     // мітка всередині пікселя
}

// [GO: PACKAGE-LEVEL VAR]
// Три типи ворогів. Визначаються один раз, читаються скрізь у пакеті.
var (
	// ConfigBoid — стадний, помірний. Колір і агресія рандомні на старті.
	ConfigBoid = EnemyConfig{
		WanderStrength:  0.2,
		AlignmentRate:   0.03,
		MaxSpeed:        0.9,
		AggressionForce: 0.04,
		BurstChance:     0.0001,
		BurstForce:      50.0,
		DetectionRange:  80.0,
		PounceMulti:     7.0,
		MaxHP:           2,
		// Color.A == 0 → aggressionColor() визначить колір
		Label: "FOE",
	}

	// ConfigPredator — повільний але смертоносний: великий радіус, сильний кидок.
	ConfigPredator = EnemyConfig{
		WanderStrength:  0.1,
		AlignmentRate:   0.01,
		MaxSpeed:        1.4,
		AggressionForce: 0.12,
		BurstChance:     0.0003,
		BurstForce:      80.0,
		DetectionRange:  160.0,
		PounceMulti:     14.0,
		MaxHP:           5,
		Color:           color.RGBA{220, 50, 50, 255},
		Label:           "PRD",
	}

	// ConfigSpeeder — хаотичний, дуже швидкий, крихкий, майже не флокується.
	ConfigSpeeder = EnemyConfig{
		WanderStrength:  0.6,
		AlignmentRate:   0.005,
		MaxSpeed:        2.8,
		AggressionForce: 0.02,
		BurstChance:     0.004,
		BurstForce:      120.0,
		DetectionRange:  40.0,
		PounceMulti:     2.0,
		MaxHP:           1,
		Color:           color.RGBA{200, 220, 50, 255},
		Label:           "SPD",
	}
)

// Pixel описує одну частинку — гравця або ворога.
type Pixel struct {
	X, Y       float32
	VX, VY     float32 // вектор швидкості
	AX, AY     float32 // вектор прискорення (alignment + chase)
	Aggression float32 // 0.0..1.0: для Boid — рандомний, для інших — 1.0
	HP, MaxHP  int
	HitTimer   int
	Color      color.RGBA
	Label      string
	Cfg        EnemyConfig // конфіг типу (порожній для гравця)
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
	configs := []EnemyConfig{ConfigBoid, ConfigPredator, ConfigSpeeder}
	enemies := make([]Pixel, count)

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

		enemies[i] = Pixel{
			X:          spawnX,
			Y:          spawnY,
			VX:         (rand.Float32() - 0.5) * cfg.MaxSpeed,
			VY:         (rand.Float32() - 0.5) * cfg.MaxSpeed,
			Aggression: aggression,
			HP:         cfg.MaxHP,
			MaxHP:      cfg.MaxHP,
			Color:      col,
			Label:      cfg.Label,
			Cfg:        cfg,
		}
	}
	return enemies
}
