package main

import (
	"math"

	"github.com/hajimehoshi/ebiten/v2"
)

// handlePlayerInput читає клавіші і додає прискорення до вектора швидкості.
// Не змінює позицію напряму — це робить updatePlayer().
//
// Дві схеми залежно від виду:
//
//	топ-даун   — стрілки рухають у СВІТОВИХ координатах (як завжди);
//	від 1-ї особи — ліво/право ПОВЕРТАЮТЬ камеру, вперед/назад рухають У БІК ПОГЛЯДУ.
func (g *Game) handlePlayerInput() {
	if g.firstPerson {
		g.handleFirstPersonInput()
		return
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowUp) || ebiten.IsKeyPressed(ebiten.KeyW) {
		g.player.VelY -= playerAccel
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowDown) || ebiten.IsKeyPressed(ebiten.KeyS) {
		g.player.VelY += playerAccel
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowLeft) || ebiten.IsKeyPressed(ebiten.KeyA) {
		g.player.VelX -= playerAccel
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowRight) || ebiten.IsKeyPressed(ebiten.KeyD) {
		g.player.VelX += playerAccel
	}
}

// handleFirstPersonInput — класичне raycaster-керування.
// Ліво/право — поворот камери; вперед/назад — рух у напрямку camAngle.
func (g *Game) handleFirstPersonInput() {
	if ebiten.IsKeyPressed(ebiten.KeyArrowLeft) || ebiten.IsKeyPressed(ebiten.KeyA) {
		g.camAngle -= playerTurnSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowRight) || ebiten.IsKeyPressed(ebiten.KeyD) {
		g.camAngle += playerTurnSpeed
	}
	dirX := float32(math.Cos(float64(g.camAngle)))
	dirY := float32(math.Sin(float64(g.camAngle)))
	if ebiten.IsKeyPressed(ebiten.KeyArrowUp) || ebiten.IsKeyPressed(ebiten.KeyW) {
		g.player.VelX += playerAccel * dirX
		g.player.VelY += playerAccel * dirY
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowDown) || ebiten.IsKeyPressed(ebiten.KeyS) {
		g.player.VelX -= playerAccel * dirX
		g.player.VelY -= playerAccel * dirY
	}
}

// updatePlayer застосовує тертя, обмежує швидкість і рухає гравця.
// Макс швидкість росте через sqrt(difficulty) — повільніше ніж вороги.
func (g *Game) updatePlayer() {
	// Тертя — при відпусканні клавіші гравець поступово зупиняється (інерція)
	g.player.VelX *= playerFriction
	g.player.VelY *= playerFriction

	// sqrt робить ріст плавнішим: 1→1.41→1.73→2.0 замість 1→2→3→4
	maxSpeed := float32(playerBaseSpeed) * float32(math.Sqrt(float64(g.difficulty)))
	speed := float32(math.Sqrt(float64(g.player.VelX*g.player.VelX + g.player.VelY*g.player.VelY)))
	if speed > maxSpeed {
		g.player.VelX = g.player.VelX / speed * maxSpeed
		g.player.VelY = g.player.VelY / speed * maxSpeed
	}

	newX := g.player.X + g.player.VelX
	newY := g.player.Y + g.player.VelY

	// Внутрішні стіни — slide (зупиняємо відповідну вісь).
	// Межі екрану — ігноруємо тут, wrap-around нижче.
	if !isInteriorWallRect(newX, g.player.Y) {
		g.player.X = newX
	} else {
		g.player.VelX = 0
	}
	if !isInteriorWallRect(g.player.X, newY) {
		g.player.Y = newY
	} else {
		g.player.VelY = 0
	}

	// Wrap-around: вилітаєш за край — з'являєшся з іншого боку.
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
