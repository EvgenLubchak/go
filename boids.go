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
				var avgX, avgY float32 // cohesion: центр маси сусідів
				var sepX, sepY float32 // separation: сума векторів відштовхування
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

				// Chase або Brain — залежить від типу ворога.
				fdx := g.player.X - e.X
				fdy := g.player.Y - e.Y
				dist := float32(math.Sqrt(float64(fdx*fdx + fdy*fdy)))

				if e.Brain != nil {
					// [RL: Q-LEARNING — агент без вчителя]
					// 1. state: куди гравець + 8 whiskers (зір на стіни)
					// 2. Step: оцінює минулу дію за reward і обирає нову (ε-greedy)
					// 3. дія = один з 8 напрямків → прискорення туди
					// e.HitWall (наслідок минулого руху, виставлений у updateEnemies)
					// стає сигналом штрафу за зіткнення зі стіною.
					// [ВБИВЦЯ] Той самий Step, але ІНШИЙ набір входів: замість
					// прямого напрямку — flow-field (шлях крізь стіни).
					var state [baseInputs]float32
					if e.Cfg.UsesFlowField {
						state = GatherKillerInputs(e, &g.player, &g.flow)
					} else {
						state = GatherInputs(e, &g.player)
					}
					action := e.Brain.Step(state, dist, e.HitWall)

					e.AccX += dirs8[action][0] * brainForce * g.difficulty
					e.AccY += dirs8[action][1] * brainForce * g.difficulty

					// [СТИГМЕРГІЯ] Відштовхування від слідів фрустрації навколо (лише
					// якщо феромони ввімкнені): рій уникає місць, де вже застрягав.
					if pheromonesEnabled {
						ffx, ffy := g.frustrationForce(e.X, e.Y)
						e.AccX += ffx
						e.AccY += ffy
					}
				} else if dist > 0 && e.Aggression > 0 {
					// Звичайні вороги: hardcoded chase з pounce
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
	g.decayFrustration() // [СТИГМЕРГІЯ] сліди тануть щокадру (однопотоково)

	for i := range g.enemies {
		e := &g.enemies[i]

		// [RL] Скидаємо прапор удару — фіксуємо зіткнення саме цього кадру.
		// calcAcceleration наступного кадру прочитає його як сигнал штрафу.
		e.HitWall = false

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
		if e.InvulnTimer > 0 { // [БІЙ] кадри невразливості після удару
			e.InvulnTimer--
		}

		// Рух: відбивання від тайлових стін і країв екрану.
		newX := e.X + e.VelX
		newY := e.Y + e.VelY

		if !isWallRect(newX, e.Y) {
			e.X = newX
		} else {
			e.VelX = -e.VelX
			e.HitWall = true
		}
		if !isWallRect(e.X, newY) {
			e.Y = newY
		} else {
			e.VelY = -e.VelY
			e.HitWall = true
		}

		// Додатковий захист від виходу за межі (якщо ворог якось вийшов)
		if e.X < 0 {
			e.X = 0
			e.VelX = -e.VelX
			e.HitWall = true
		}
		if e.X > screenWidth-pixelSize {
			e.X = screenWidth - pixelSize
			e.VelX = -e.VelX
			e.HitWall = true
		}
		if e.Y < 0 {
			e.Y = 0
			e.VelY = -e.VelY
			e.HitWall = true
		}
		if e.Y > screenHeight-pixelSize {
			e.Y = screenHeight - pixelSize
			e.VelY = -e.VelY
			e.HitWall = true
		}

		// [СТИГМЕРГІЯ] Учень ОФІЦІЙНО застряг (спрацювала фрустрація) → лишаємо слід
		// саме в цій клітинці (лише якщо феромони ввімкнені). НЕ на кожен дотик
		// стіни (інакше «слимачий слід»), а лише в реальних глухих кутах.
		if pheromonesEnabled && e.Brain != nil && e.Brain.markStuck {
			e.Brain.markStuck = false
			cx := int(e.X) / pixelSize
			cy := int(e.Y) / pixelSize
			if cx >= 0 && cx < boidMapW && cy >= 0 && cy < boidMapH {
				g.frustration[cy][cx] += frustrationDeposit
			}
		}
	}
}

// decayFrustration притлумлює всі сліди феромонів (однопотоково, щокадру).
// Завдяки затуханню «погане місце» з часом забувається й стає прохідним знову.
func (g *Game) decayFrustration() {
	for y := range g.frustration {
		for x := range g.frustration[y] {
			g.frustration[y][x] *= frustrationDecay
		}
	}
}

// frustrationForce — сила відштовхування від слідів феромонів навколо точки.
// Сумуємо вектори «від клітинки-сліду до агента», зважені силою сліду й поділені
// на відстань (ближчий слід штовхає сильніше). Лише ЧИТАННЯ сітки → безпечно
// в паралельному calcAcceleration (запис відбувається в окремій фазі).
func (g *Game) frustrationForce(x, y float32) (fx, fy float32) {
	cx := int(x) / pixelSize
	cy := int(y) / pixelSize
	for dy := -frustrationRadius; dy <= frustrationRadius; dy++ {
		for dx := -frustrationRadius; dx <= frustrationRadius; dx++ {
			nx, ny := cx+dx, cy+dy
			if nx < 0 || nx >= boidMapW || ny < 0 || ny >= boidMapH {
				continue
			}
			f := g.frustration[ny][nx]
			if f <= 0 {
				continue
			}
			rx := x - (float32(nx)*pixelSize + pixelSize/2)
			ry := y - (float32(ny)*pixelSize + pixelSize/2)
			d := float32(math.Sqrt(float64(rx*rx + ry*ry)))
			if d < 1 {
				d = 1
			}
			fx += rx / d * f
			fy += ry / d * f
		}
	}
	return fx * frustrationRepel, fy * frustrationRepel
}
