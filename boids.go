package main

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// updateBoidMap очищає сітку і розставляє ворогів.
func (g *Game) updateBoidMap() {
	for y := range g.boidMap {
		for x := range g.boidMap[y] {
			g.boidMap[y][x] = 0
		}
	}
	for i, e := range g.enemies {
		cx := int(e.X) / pixelSize
		cy := int(e.Y) / pixelSize
		if cx >= 0 && cx < boidMapW && cy >= 0 && cy < boidMapH {
			g.boidMap[cy][cx] = i + 1
		}
	}
}

// calcAcceleration рахує alignment (boids) і chase для кожного ворога.
//
// [GO: GOROUTINES + SYNC.WAITGROUP]
// Розбиваємо ворогів на chunks і обраховуємо кожен у окремому goroutine.
// WaitGroup лічить активні goroutines: Add(1) перед запуском, Done() всередині,
// Wait() блокує поки всі не завершились.
//
// Worker pool: runtime.NumCPU() goroutines замість одного на кожного ворога —
// мінімальний overhead при максимальному паралелізмі.
func (g *Game) calcAcceleration() {
	n := len(g.enemies)
	if n == 0 {
		return
	}

	// [GO: SNAPSHOT PATTERN]
	// Копіюємо VX/VY всіх ворогів перед паралельним обрахунком.
	// Goroutines читають snapshot (незмінний) → пишуть тільки у свій AX/AY.
	// Без snapshot: одна goroutine читала б VX сусіда поки інша пише його AX
	// (різні поля struct, але Go race detector це все одно помічає).
	type vel struct{ VX, VY float32 }
	vels := make([]vel, n)
	for i := range g.enemies {
		vels[i] = vel{g.enemies[i].VX, g.enemies[i].VY}
	}

	// Ділимо ворогів рівномірно між CPU ядрами
	numWorkers := runtime.NumCPU()
	chunkSize := (n + numWorkers - 1) / numWorkers // округлення вгору

	// [GO: SYNC.WAITGROUP]
	// var wg sync.WaitGroup — лічильник goroutines.
	// wg.Add(1) перед go func → wg.Done() при завершенні → wg.Wait() чекає всіх.
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}

		wg.Add(1)

		// [GO: GO FUNC з параметрами]
		// start і end передаємо як аргументи — інакше замикання захопить змінну
		// по посиланню і всі goroutines побачать одне й те саме значення на момент запуску.
		go func(start, end int) {
			defer wg.Done() // [GO: DEFER] — гарантовано викличеться при виході з функції

			for i := start; i < end; i++ {
				e := &g.enemies[i]
				cx := int(e.X) / pixelSize
				cy := int(e.Y) / pixelSize

				var avgVX, avgVY float32
				count := 0

				for dy := -visionRadius; dy <= visionRadius; dy++ {
					for dx := -visionRadius; dx <= visionRadius; dx++ {
						nx, ny := cx+dx, cy+dy
						if nx < 0 || nx >= boidMapW || ny < 0 || ny >= boidMapH {
							continue
						}
						idx := g.boidMap[ny][nx]
						if idx == 0 || idx-1 == i {
							continue
						}
						// Читаємо з snapshot — race-free
						avgVX += vels[idx-1].VX
						avgVY += vels[idx-1].VY
						count++
					}
				}

				if count > 0 {
					avgVX /= float32(count)
					avgVY /= float32(count)
					e.AX = (avgVX - e.VX) * e.Cfg.AlignmentRate
					e.AY = (avgVY - e.VY) * e.Cfg.AlignmentRate
				} else {
					e.AX = 0
					e.AY = 0
				}

				// Chase: пишемо тільки у g.enemies[i] — виключно наш chunk
				fdx := g.player.X - e.X
				fdy := g.player.Y - e.Y
				dist := float32(math.Sqrt(float64(fdx*fdx + fdy*fdy)))
				if dist > 0 && e.Aggression > 0 {
					if dist < e.Cfg.DetectionRange {
						pounce := (1 - dist/e.Cfg.DetectionRange) * e.Cfg.PounceMulti
						e.AX += (fdx / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
						e.AY += (fdy / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
					}
				}
			}
		}(start, end)
	}

	// [GO: WAWG.WAIT]
	// Блокуємо головний goroutine поки всі workers не завершать свій chunk.
	// Тільки після цього updateEnemies() отримає актуальні AX/AY.
	wg.Wait()
}

// updateEnemies застосовує блукання, burst, прискорення, damping, рух і відбивання.
func (g *Game) updateEnemies() {
	for i := range g.enemies {
		e := &g.enemies[i]

		// [GO: e.Cfg.WanderStrength] — Speeder блукає хаотично, Predator — плавно
		e.VX += (rand.Float32() - 0.5) * e.Cfg.WanderStrength
		e.VY += (rand.Float32() - 0.5) * e.Cfg.WanderStrength

		// Burst: Speeder б'є часто і сильно, Boid — рідко і слабко
		if rand.Float32() < e.Cfg.BurstChance {
			angle := rand.Float64() * 2 * math.Pi
			e.VX += float32(math.Cos(angle)) * e.Cfg.BurstForce
			e.VY += float32(math.Sin(angle)) * e.Cfg.BurstForce
		}

		e.VX += e.AX
		e.VY += e.AY

		e.VX *= damping
		e.VY *= damping

		// [GO: e.Cfg.MaxSpeed] — стеля швидкості своя у кожного типу
		currentMaxSpeed := e.Cfg.MaxSpeed * g.difficulty
		speed := float32(math.Sqrt(float64(e.VX*e.VX + e.VY*e.VY)))
		if speed > currentMaxSpeed {
			e.VX = e.VX / speed * currentMaxSpeed
			e.VY = e.VY / speed * currentMaxSpeed
		}

		if e.HitTimer > 0 {
			e.HitTimer--
		}

		// Рух: відбивання від тайлових стін і країв екрану.
		newX := e.X + e.VX
		newY := e.Y + e.VY

		if !isWallRect(newX, e.Y) {
			e.X = newX
		} else {
			e.VX = -e.VX
		}
		if !isWallRect(e.X, newY) {
			e.Y = newY
		} else {
			e.VY = -e.VY
		}

		// Додатковий захист від виходу за межі (якщо ворог якось вийшов)
		if e.X < 0 {
			e.X = 0
			e.VX = -e.VX
		}
		if e.X > screenWidth-pixelSize {
			e.X = screenWidth - pixelSize
			e.VX = -e.VX
		}
		if e.Y < 0 {
			e.Y = 0
			e.VY = -e.VY
		}
		if e.Y > screenHeight-pixelSize {
			e.Y = screenHeight - pixelSize
			e.VY = -e.VY
		}
	}
}
