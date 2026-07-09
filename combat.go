package main

import (
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

// collides перевіряє чи перетинаються два квадрати (AABB collision).
func collides(ax, ay, bx, by float32) bool {
	return ax < bx+pixelSize &&
		ax+pixelSize > bx &&
		ay < by+pixelSize &&
		ay+pixelSize > by
}

// checkCollisions перевіряє чи торкнувся ворог гравця → game over.
func (g *Game) checkCollisions() {
	for _, e := range g.enemies {
		if collides(g.player.X, g.player.Y, e.X, e.Y) {
			g.metrics.catches++ // [МЕТРИКИ] спіймання (для catch-rate)
			if aiPlayer {
				// [SELF-PLAY] Не game over — переносимо жертву й тренуємось далі.
				g.respawnPlayer()
			} else {
				g.gameOver = true
				g.saveBrains() // зберігаємо мозок хижаків
			}
			return
		}
	}
}

// playerAttack обробляє удар SPACE: cooldown, пошкодження в радіусі, анімація.
// [GO: inpututil.IsKeyJustPressed] — спрацьовує тільки в перший кадр натискання.
// ebiten.IsKeyPressed спрацьовував би кожен кадр поки клавіша утримується.
func (g *Game) playerAttack() {
	if !inpututil.IsKeyJustPressed(ebiten.KeySpace) || g.attackCooldown > 0 {
		return
	}
	g.attackCooldown = attackCooldownMax
	g.attackTimer = attackDuration

	px := g.player.X + pixelSize/2
	py := g.player.Y + pixelSize/2

	for i := range g.enemies {
		e := &g.enemies[i]
		ex := e.X + pixelSize/2
		ey := e.Y + pixelSize/2
		dx := px - ex
		dy := py - ey
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
		if dist <= attackRadius {
			e.HP -= attackDamage
			e.HitTimer = hitFlashDuration
		}
	}
}

// removeDeadEnemies видаляє ворогів з HP <= 0.
// [GO: FILTER SLICE in-place]
// g.enemies[:0] — той самий масив у пам'яті, але довжина 0.
// append пише поверх — без нової алокації пам'яті.
func (g *Game) removeDeadEnemies() {
	alive := g.enemies[:0]
	for _, e := range g.enemies {
		if e.HP > 0 {
			alive = append(alive, e)
		}
	}
	g.enemies = alive
}
