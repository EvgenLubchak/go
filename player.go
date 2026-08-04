package main

import (
	"math"
	"math/rand"

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

// updatePrey — [SELF-PLAY] гравцем керує мозок-жертва (замість клавіатури).
// Спостерігає загрозу, обирає дію (тікати), прискорюється в той бік. Далі
// updatePlayer застосує фізику (тертя, стіни, обмеження швидкості).
func (g *Game) updatePrey() {
	state := GatherPreyInputs(&g.player, g.units)
	action := g.player.Brain.Step(state, g.player.HitWall)
	g.player.VelX += dirs8[action][0] * playerAccel
	g.player.VelY += dirs8[action][1] * playerAccel
}

// respawnPlayer — [SELF-PLAY] після спіймання переносимо гравця у випадкове
// відкрите місце й продовжуємо тренування (без game over). Скидаємо hasPrev,
// щоб мозок-жертва не вчився на «телепорті».
func (g *Game) respawnPlayer() {
	for tries := 0; tries < 50; tries++ {
		x := float32(rand.Intn(screenWidth - pixelSize))
		y := float32(rand.Intn(screenHeight - pixelSize))
		if !isInteriorWallRect(x, y) {
			g.player.X, g.player.Y = x, y
			break
		}
	}
	g.player.VelX, g.player.VelY = 0, 0
	g.player.HP = playerMaxHP // [БІЙ] новий «епізод» → повне здоровʼя
	g.player.resetFur()       // [ВОРС] інакше хутро «прилетіло б» зі старого місця
	g.player.InvulnTimer = 0
	if g.player.Brain != nil {
		g.player.Brain.hasPrev = false
		g.player.Brain.h = [gruHidden]float32{} // [RNN] скидаємо рекурентну памʼять
		g.player.Brain.seqN = 0                 // [RNN] відкидаємо недособраний відрізок
		g.player.Brain.progress = 0             // [RL] новий епізод — прогресу ще нема
	}
}

// updatePlayer застосовує тертя, обмежує швидкість і рухає гравця.
// Макс швидкість росте через sqrt(difficulty) — повільніше ніж вороги.
func (g *Game) updatePlayer() {
	g.player.HitWall = false // [SELF-PLAY] сигнал удару об стіну для мозку-жертви

	// [БІЙ] Тікають кадри невразливості й білого блимання після удару.
	if g.player.InvulnTimer > 0 {
		g.player.InvulnTimer--
	}
	if g.player.HitTimer > 0 {
		g.player.HitTimer--
	}

	// Тертя — при відпусканні клавіші гравець поступово зупиняється (інерція)
	g.player.VelX *= playerFriction
	g.player.VelY *= playerFriction

	// sqrt робить ріст плавнішим: 1→1.41→1.73→2.0 замість 1→2→3→4
	maxSpeed := float32(playerBaseSpeed) * float32(math.Sqrt(float64(g.difficulty)))
	// [БІЙ] Гравця теж відкидає ударом — інакше удар по ньому не відчувався б.
	// Тертя гравця (0.90) гасить віддачу швидше, ніж у юнітів (0.95), тож контроль
	// повертається за кілька кадрів.
	if g.player.KnockTimer > 0 {
		g.player.KnockTimer--
		maxSpeed *= knockSpeedMulti
	}
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
		g.player.HitWall = true
	}
	if !isInteriorWallRect(g.player.X, newY) {
		g.player.Y = newY
	} else {
		g.player.VelY = 0
		g.player.HitWall = true
	}

	// Межа екрану — суцільна стіна (без телепорту): зупиняємось на краю й
	// сигналимо HitWall, так само як від внутрішніх стін. Вороги від межі
	// відбиваються — тепер гравець теж не «протікає» наскрізь.
	if g.player.X < 0 {
		g.player.X = 0
		g.player.VelX = 0
		g.player.HitWall = true
	}
	if g.player.X > screenWidth-pixelSize {
		g.player.X = screenWidth - pixelSize
		g.player.VelX = 0
		g.player.HitWall = true
	}
	if g.player.Y < 0 {
		g.player.Y = 0
		g.player.VelY = 0
		g.player.HitWall = true
	}
	if g.player.Y > screenHeight-pixelSize {
		g.player.Y = screenHeight - pixelSize
		g.player.VelY = 0
		g.player.HitWall = true
	}

	// [ВОРС] Гравець НЕ входить у g.units, тож updateUnits його не чіпає —
	// оновлюємо хутро тут, у самому кінці (позиція вже остаточна). Без цього
	// ворс завис би на місці спавну, а смуги розтяглись би через пів карти.
	updateFur(&g.player)
	updateBody(&g.player) // [ТІЛО] у гравця немає мозку → лише пружина, без Q-форми
}
