package main

import (
	"image/color"
	"math"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
)

// newPlayer — гравець на старті гри.
//
// Окремою функцією, а не літералом у main, з тієї ж причини, що ballHome і tentJointsOf:
// інакше налаштування гравця нічим не перевірити. Літерал усередині main() недосяжний
// для тесту, тож рядок `Cfg: ...` можна було б видалити й нічого б не впало —
// а це саме той тихий розʼїзд, від якого решта кріплень уже застрахована.
//
// [КОНФІГ ГРАВЦЯ] Cfg у гравця лишається майже порожнім і надалі: він живе поза
// unitRoster, і на порожні поля спираються updateBody (MaxSpeed ≤ 0 → playerBaseSpeed)
// та GatherInputs. Заповнюємо РІВНО одне поле — те, для якого в гравця є своя ручка.
func newPlayer() Pixel {
	return Pixel{
		X: playerSpawn.X,
		Y: playerSpawn.Y,
		// [БІЙ] HP > 0 → гравець витримує кілька ударів; MaxHP>0 ще й вмикає
		// малювання HP-бару в drawPixel (те саме, що у ворогів).
		HP:      playerMaxHP,
		MaxHP:   playerMaxHP,
		Faction: factionPlayer,
		Color:   color.RGBA{R: 0, G: 255, B: 180, A: 255},

		// [ВІДРОСТОК] Нуль тут означає «взяти дефолт tentJoints» — саме так гравець і
		// жив досі, просто неявно. Тепер це видно й керовано з tuning_visual.go.
		Cfg: UnitConfig{TentJoints: playerTentJoints},
	}
}

// handlePlayerInput читає клавіші і додає прискорення до вектора швидкості.
// Не змінює позицію напряму — це робить updatePlayer().
//
// Дві схеми залежно від виду:
//
//	топ-даун   — стрілки рухають у СВІТОВИХ координатах (як завжди);
//	від 1-ї особи — ліво/право ПОВЕРТАЮТЬ камеру, вперед/назад рухають У БІК ПОГЛЯДУ.
func (g *Game) handlePlayerInput() {
	// [РИВОК] У будь-якій фазі атаки керування заблоковане — у цьому і є ЦІНА удару.
	// Замах видно, напрямок замкнено, відхід безпорадний: саме звідси в юніта
	// зʼявляється що читати й коли карати.
	if g.player.DashPhase != dashIdle {
		return
	}
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
	// [УХИЛЕННЯ] Девʼята дія без напрямку — те саме правило, що в юнітів.
	if action == actionDodge {
		// Через canDodge, а не через власну перевірку кулдауну: правило «почав —
		// доводь» мусить бути одне на всіх. З окремою умовою жертва могла б ухилятись
		// просто з відходу — точнісінько як юніти до виправлення.
		//
		// ⚠️ ВІДОМА АСИМЕТРІЯ: кидка (dodgeBurst) у жертви немає — лише невразливість.
		// Це не вада цього виправлення, а незакінчена механіка: перш ніж її додавати,
		// треба вирішити, від ЧОГО жертва відскакує, коли загроз кілька.
		if canDodge(&g.player) {
			g.player.DodgeTimer = dodgeInvuln
			g.player.DodgeCooldown = dodgeCooldown
		}
		return
	}
	g.player.VelX += dirs8[action][0] * playerAccel
	g.player.VelY += dirs8[action][1] * playerAccel
}

// respawnPlayer — [SELF-PLAY] після спіймання переносимо гравця у випадкове
// відкрите місце й продовжуємо тренування (без game over). Скидаємо hasPrev,
// щоб мозок-жертва не вчився на «телепорті».
func (g *Game) respawnPlayer() {
	for tries := 0; tries < 50; tries++ {
		x := float32(rand.Intn(worldWidth - pixelSize))
		y := float32(rand.Intn(worldHeight - pixelSize))
		if !isInteriorWallRect(x, y) {
			g.player.X, g.player.Y = x, y
			break
		}
	}
	g.player.VelX, g.player.VelY = 0, 0
	g.player.HP = playerMaxHP // [БІЙ] новий «епізод» → повне здоровʼя
	g.player.resetFur()       // [ВОРС] інакше хутро «прилетіло б» зі старого місця
	g.player.InvulnTimer = 0
	// [SELF-PLAY] Інакше жертва прийшла б у нове життя з недотіклим ухиленням, і
	// «новий епізод» починався б із подарованої невразливості.
	g.player.DodgeTimer, g.player.DodgeCooldown, g.player.DodgeRecover = 0, 0, 0
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
	if g.player.StrikeTimer > 0 {
		g.player.StrikeTimer--
	}
	if g.player.HitTimer > 0 {
		g.player.HitTimer--
	}

	// [SELF-PLAY] ⚠️ ТАЙМЕРИ УХИЛЕННЯ ГРАВЦЯ ТІКАЮТЬ САМЕ ТУТ, І ЦЕ НЕ ДРІБНИЦЯ.
	//
	// Гравець НЕ входить у g.units, тож цикл таймерів в updateUnits його не бачить —
	// та сама пастка, що колись підвісила йому ворс на місці спавну. А updatePrey
	// (режим aiPlayer) виставляє DodgeTimer при дії №8.
	//
	// Без цих рядків таймер лишався 20 НАЗАВЖДИ, а applyImpactDamage починається з
	// «якщо DodgeTimer > 0 — шкоди немає». Тобто жертва ставала невразливою після
	// першого ж ухилення, рій фізично не міг її спіймати, і вся арена self-play тихо
	// вироджувалась: спіймань немає → нагороди немає → «хижаки не навчились».
	//
	// Гра при цьому не падала й нічого не показувала. Найгірший вид вади.
	//
	// Порядок повторює цикл юнітів дослівно, включно зі стиком невразливості й
	// відходу впритул — інакше в жертви лишився б БЕЗКОШТОВНИЙ дож, тобто рівно та
	// домінантна дія, яку ми щойно вилікували в стражників.
	if g.player.DodgeTimer > 0 {
		g.player.DodgeTimer--
		if g.player.DodgeTimer == 0 {
			g.player.DodgeRecover = dodgeRecovery
		}
	} else if g.player.DodgeRecover > 0 {
		g.player.DodgeRecover--
	}
	if g.player.DodgeCooldown > 0 {
		g.player.DodgeCooldown--
	}

	// [РИВОК] Машина фаз — ДО тертя й до стелі швидкості: у замаху ми швидкість
	// зануляємо, у ривку задаємо, і тертя не має цього псувати.
	advanceDash(&g.player)

	// Тертя — при відпусканні клавіші гравець поступово зупиняється (інерція)
	g.player.VelX *= playerFriction
	g.player.VelY *= playerFriction

	// sqrt робить ріст плавнішим: 1→1.41→1.73→2.0 замість 1→2→3→4
	base := float32(playerBaseSpeed) * float32(math.Sqrt(float64(g.difficulty)))
	maxSpeed := base
	// [БІЙ] Гравця теж відкидає ударом — інакше удар по ньому не відчувався б.
	// Тертя гравця (0.90) гасить віддачу швидше, ніж у юнітів (0.95), тож контроль
	// повертається за кілька кадрів.
	if g.player.KnockTimer > 0 {
		g.player.KnockTimer--
		maxSpeed *= knockSpeedMulti
	}
	// [РИВОК] Замах — ПОВНА зупинка: це і є телеграф, і він мусить бути безсумнівним.
	// Ривок — рух по замкненій лінії з піднятою стелею (10 кадрів × 10px = 100px,
	// чотири корпуси). Відхід нічого не задає: гравця несе за інерцією й гасить тертя.
	switch g.player.DashPhase {
	case dashPhaseWindup:
		g.player.VelX, g.player.VelY = 0, 0
	case dashPhaseActive:
		// Швидкість ривка — від БАЗОВОЇ стелі, а НЕ від піднятої віддачею. Інакше удар,
		// що прилетів під час ривка, множив би стелю двічі: playerBaseSpeed 5.0 ×
		// knockSpeedMulti 4.0 × dashSpeedMulti 2.0 = 40px/кадр — телепорт через пів
		// екрана замість ривка. Дві незалежні механіки підняття стелі не мусять
		// множитись; беремо більшу з двох.
		//
		// (Число перераховано: knockSpeedMulti з того часу став 4.0 замість 5.0, і
		// добуток тут стояв старий. Висновок не змінився, арифметика — так.)
		ds := base * dashSpeedMulti
		g.player.VelX = g.player.DashDirX * ds
		g.player.VelY = g.player.DashDirY * ds
		if ds > maxSpeed {
			maxSpeed = ds
		}
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
	if g.player.X > worldWidth-pixelSize {
		g.player.X = worldWidth - pixelSize
		g.player.VelX = 0
		g.player.HitWall = true
	}
	if g.player.Y < 0 {
		g.player.Y = 0
		g.player.VelY = 0
		g.player.HitWall = true
	}
	if g.player.Y > worldHeight-pixelSize {
		g.player.Y = worldHeight - pixelSize
		g.player.VelY = 0
		g.player.HitWall = true
	}

	// [ВОРС] Гравець НЕ входить у g.units, тож updateUnits його не чіпає —
	// оновлюємо хутро тут, у самому кінці (позиція вже остаточна). Без цього
	// ворс завис би на місці спавну, а смуги розтяглись би через пів карти.
	updateFur(&g.player)
	updateBalls(&g.player)
	updateTentacle(&g.player)
	updateLimbs(&g.player)
	updateFace(&g.player)
	updateBody(&g.player) // [ТІЛО] у гравця немає мозку → лише пружина, без Q-форми
}
