package main

import (
	"math"
	"math/rand"
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

// calcAcceleration рахує alignment (boids) і chase (хижак) для кожного ворога.
// Кожен ворог використовує свій Cfg — різні типи поводяться по-різному.
func (g *Game) calcAcceleration() {
	for i := range g.enemies {
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
				neighbor := g.enemies[idx-1]
				avgVX += neighbor.VX
				avgVY += neighbor.VY
				count++
			}
		}

		if count > 0 {
			avgVX /= float32(count)
			avgVY /= float32(count)
			// [GO: e.Cfg.AlignmentRate] — кожен тип має власну силу флокування
			e.AX = (avgVX - e.VX) * e.Cfg.AlignmentRate
			e.AY = (avgVY - e.VY) * e.Cfg.AlignmentRate
		} else {
			e.AX = 0
			e.AY = 0
		}

		// Chase: хижацький кидок до гравця.
		// Predator: великий DetectionRange + PounceMulti → смертоносний на дистанції.
		// Speeder: малий DetectionRange → майже ігнорує гравця здалеку.
		dx := g.player.X - e.X
		dy := g.player.Y - e.Y
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
		if dist > 0 && e.Aggression > 0 {
			if dist < e.Cfg.DetectionRange {
				pounce := (1 - dist/e.Cfg.DetectionRange) * e.Cfg.PounceMulti
				e.AX += (dx / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
				e.AY += (dy / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
			}
		}
	}
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

		e.X += e.VX
		e.Y += e.VY

		if e.X <= 0 {
			e.X = 0
			e.VX = -e.VX
		}
		if e.X >= screenWidth-pixelSize {
			e.X = screenWidth - pixelSize
			e.VX = -e.VX
		}
		if e.Y <= 0 {
			e.Y = 0
			e.VY = -e.VY
		}
		if e.Y >= screenHeight-pixelSize {
			e.Y = screenHeight - pixelSize
			e.VY = -e.VY
		}
	}
}
