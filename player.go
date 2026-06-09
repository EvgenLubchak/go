package main

import (
	"math"

	"github.com/hajimehoshi/ebiten/v2"
)

// handlePlayerInput читає клавіші і додає прискорення до вектора швидкості.
// Не змінює позицію напряму — це робить updatePlayer().
func (g *Game) handlePlayerInput() {
	if ebiten.IsKeyPressed(ebiten.KeyArrowUp) || ebiten.IsKeyPressed(ebiten.KeyW) {
		g.player.VY -= playerAccel
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowDown) || ebiten.IsKeyPressed(ebiten.KeyS) {
		g.player.VY += playerAccel
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowLeft) || ebiten.IsKeyPressed(ebiten.KeyA) {
		g.player.VX -= playerAccel
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowRight) || ebiten.IsKeyPressed(ebiten.KeyD) {
		g.player.VX += playerAccel
	}
}

// updatePlayer застосовує тертя, обмежує швидкість і рухає гравця.
// Макс швидкість росте через sqrt(difficulty) — повільніше ніж вороги.
func (g *Game) updatePlayer() {
	// Тертя — при відпусканні клавіші гравець поступово зупиняється (інерція)
	g.player.VX *= playerFriction
	g.player.VY *= playerFriction

	// sqrt робить ріст плавнішим: 1→1.41→1.73→2.0 замість 1→2→3→4
	maxSpeed := float32(playerBaseSpeed) * float32(math.Sqrt(float64(g.difficulty)))
	speed := float32(math.Sqrt(float64(g.player.VX*g.player.VX + g.player.VY*g.player.VY)))
	if speed > maxSpeed {
		g.player.VX = g.player.VX / speed * maxSpeed
		g.player.VY = g.player.VY / speed * maxSpeed
	}

	g.player.X += g.player.VX
	g.player.Y += g.player.VY

	// Wrap-around: вилітаєш за край — з'являєшся з іншого боку
	if g.player.X > screenWidth {
		g.player.X = 0
	}
	if g.player.X < -pixelSize {
		g.player.X = screenWidth
	}
	if g.player.Y > screenHeight {
		g.player.Y = 0
	}
	if g.player.Y < -pixelSize {
		g.player.Y = screenHeight
	}
}
