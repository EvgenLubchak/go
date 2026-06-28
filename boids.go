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
	// Копіюємо VelX/VelY/X/Y всіх ворогів перед паралельним обрахунком.
	// Goroutines читають snapshot (незмінний) → пишуть тільки у свій AccX/AccY.
	// X/Y потрібні для cohesion: середня позиція сусідів (центр маси).
	type snap struct{ VelX, VelY, X, Y float32 }
	snaps := make([]snap, n)
	for i := range g.enemies {
		snaps[i] = snap{g.enemies[i].VelX, g.enemies[i].VelY, g.enemies[i].X, g.enemies[i].Y}
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
				var avgX, avgY float32   // cohesion: центр маси сусідів
				var sepX, sepY float32   // separation: сума векторів відштовхування
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
						avgVX += snaps[idx-1].VelX
						avgVY += snaps[idx-1].VelY
						avgX += snaps[idx-1].X
						avgY += snaps[idx-1].Y

						// [GO: SEPARATION]
						// Вектор від сусіда до мене (repulsion direction).
						// Ділимо на відстань: ближчий сусід = сильніше відштовхування.
						rdx := e.X - snaps[idx-1].X
						rdy := e.Y - snaps[idx-1].Y
						d := float32(math.Sqrt(float64(rdx*rdx + rdy*rdy)))
						if d > 0 {
							sepX += rdx / d
							sepY += rdy / d
						}

						count++
					}
				}

				if count > 0 {
					fc := float32(count)

					// Alignment: тягнемо швидкість до середньої швидкості сусідів
					e.AccX = (avgVX/fc - e.VelX) * e.Cfg.AlignmentRate
					e.AccY = (avgVY/fc - e.VelY) * e.Cfg.AlignmentRate

					// Cohesion: тягнемо до центру маси (одна сила до середньої позиції)
					e.AccX += (avgX/fc - e.X) * e.Cfg.CohesionRate
					e.AccY += (avgY/fc - e.Y) * e.Cfg.CohesionRate

					// Separation: відштовхуємось від кожного сусіда окремо
					// sum(repulsion/dist) — не ділимо на count, бо сума, а не середнє
					e.AccX += sepX * e.Cfg.SeparationRate
					e.AccY += sepY * e.Cfg.SeparationRate
				} else {
					e.AccX = 0
					e.AccY = 0
				}

				// Chase: пишемо тільки у g.enemies[i] — виключно наш chunk
				fdx := g.player.X - e.X
				fdy := g.player.Y - e.Y
				dist := float32(math.Sqrt(float64(fdx*fdx + fdy*fdy)))
				if dist > 0 && e.Aggression > 0 {
					if dist < e.Cfg.DetectionRange {
						pounce := (1 - dist/e.Cfg.DetectionRange) * e.Cfg.PounceMulti
						e.AccX += (fdx / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
						e.AccY += (fdy / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
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
		e.VelX += (rand.Float32() - 0.5) * e.Cfg.WanderStrength
		e.VelY += (rand.Float32() - 0.5) * e.Cfg.WanderStrength

		// Burst: Speeder б'є часто і сильно, Boid — рідко і слабко
		if rand.Float32() < e.Cfg.BurstChance {
			angle := rand.Float64() * 2 * math.Pi
			e.VelX += float32(math.Cos(angle)) * e.Cfg.BurstForce
			e.VelY += float32(math.Sin(angle)) * e.Cfg.BurstForce
		}

		e.VelX += e.AccX
		e.VelY += e.AccY

		e.VelX *= damping
		e.VelY *= damping

		// [GO: e.Cfg.MaxSpeed] — стеля швидкості своя у кожного типу
		currentMaxSpeed := e.Cfg.MaxSpeed * g.difficulty
		speed := float32(math.Sqrt(float64(e.VelX*e.VelX + e.VelY*e.VelY)))
		if speed > currentMaxSpeed {
			e.VelX = e.VelX / speed * currentMaxSpeed
			e.VelY = e.VelY / speed * currentMaxSpeed
		}

		if e.HitTimer > 0 {
			e.HitTimer--
		}

		// Рух: відбивання від тайлових стін і країв екрану.
		newX := e.X + e.VelX
		newY := e.Y + e.VelY

		if !isWallRect(newX, e.Y) {
			e.X = newX
		} else {
			e.VelX = -e.VelX
		}
		if !isWallRect(e.X, newY) {
			e.Y = newY
		} else {
			e.VelY = -e.VelY
		}

		// Додатковий захист від виходу за межі (якщо ворог якось вийшов)
		if e.X < 0 {
			e.X = 0
			e.VelX = -e.VelX
		}
		if e.X > screenWidth-pixelSize {
			e.X = screenWidth - pixelSize
			e.VelX = -e.VelX
		}
		if e.Y < 0 {
			e.Y = 0
			e.VelY = -e.VelY
		}
		if e.Y > screenHeight-pixelSize {
			e.Y = screenHeight - pixelSize
			e.VelY = -e.VelY
		}
	}
}
