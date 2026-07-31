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

// ==========================================================================
// [БІЙ] ШКОДА ВІД УДАРУ НА ШВИДКОСТІ («кидок кобри»)
//
// Шкодить не дотик, а ЗБЛИЖЕННЯ на швидкості. Міра — closing speed: проєкція
// швидкості на напрямок до цілі (звичайний dot product одиничних векторів).
//
//	повільно зіштовхнулись   → impact малий → шкоди НЕМА (нема взаємного знищення)
//	налетів на нерухомого    → шкода лише жертві (перевага атакуючому)
//	лоб-у-лоб на швидкості   → обидва отримали (реальний ризик атаки)
//	летять паралельно, торк. → impact ≈ 0 → нічого
//
// Наслідок: hit-and-run стає оптимальним МАТЕМАТИЧНО, а не бо ми так закодували.
// ==========================================================================

// closingSpeed — наскільки швидко об'єкт зі швидкістю (vx,vy) зближується з ціллю
// у напрямку (nx,ny). Додатне = летить У ціль, від'ємне = віддаляється.
func closingSpeed(vx, vy, nx, ny float32) float32 {
	return vx*nx + vy*ny
}

// unitTo — одиничний вектор від центру (ax,ay) до центру (bx,by) + чи він валідний.
func unitTo(ax, ay, bx, by float32) (nx, ny float32, ok bool) {
	dx := (bx + pixelSize/2) - (ax + pixelSize/2)
	dy := (by + pixelSize/2) - (ay + pixelSize/2)
	d := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if d < 0.001 {
		return 0, 0, false // центри збіглись — напрямок невизначений
	}
	return dx / d, dy / d, true
}

// impactThreshold — мінімальна швидкість зближення, щоб удар зарахувався:
// частка ВЛАСНОГО максимуму (щоб швидкі й повільні юніти були в рівних умовах)
// але не нижче абсолютної підлоги.
func impactThreshold(ownMaxSpeed float32) float32 {
	t := ownMaxSpeed * impactSpeedFrac
	if t < impactMinSpeed {
		t = impactMinSpeed
	}
	return t
}

// applyImpactDamage завдає шкоди, якщо ціль не в невразливості.
// Працює однаково для гравця і для ворога (обидва — Pixel).
func applyImpactDamage(p *Pixel) {
	if p.InvulnTimer > 0 {
		return
	}
	p.HP -= impactDamage
	p.HitTimer = hitFlashDuration // біле блимання (вже було для удару гравця)
	p.InvulnTimer = impactInvuln
}

// resolveImpacts — [БІЙ] проходить пари, що перетинаються, і завдає шкоди за
// closing speed. Обробляє і гравець↔вороги, і вороги↔вороги.
// Однопотоково (після паралельної фази) → без гонок.
func (g *Game) resolveImpacts() {
	// Максимальна швидкість гравця — та сама формула, що в updatePlayer.
	playerMax := float32(playerBaseSpeed) * float32(math.Sqrt(float64(g.difficulty)))

	// --- Гравець ↔ вороги ---
	for i := range g.enemies {
		e := &g.enemies[i]
		if e.HP <= 0 || !collides(g.player.X, g.player.Y, e.X, e.Y) {
			continue
		}
		if !friendlyFire && e.Faction == g.player.Faction {
			continue
		}
		nx, ny, ok := unitTo(g.player.X, g.player.Y, e.X, e.Y) // від гравця до ворога
		if !ok {
			continue
		}
		// Гравець таранить ворога.
		if closingSpeed(g.player.VelX, g.player.VelY, nx, ny) >= impactThreshold(playerMax) {
			applyImpactDamage(e)
		}
		// Ворог кидається на гравця (напрямок навпаки).
		eMax := e.Cfg.MaxSpeed * g.difficulty
		if eMax > 0 && closingSpeed(e.VelX, e.VelY, -nx, -ny) >= impactThreshold(eMax) {
			applyImpactDamage(&g.player)
		}
	}

	// --- Вороги ↔ вороги (i<j, щоб кожну пару рахувати раз) ---
	for i := range g.enemies {
		a := &g.enemies[i]
		if a.HP <= 0 {
			continue
		}
		for j := i + 1; j < len(g.enemies); j++ {
			b := &g.enemies[j]
			if b.HP <= 0 || !collides(a.X, a.Y, b.X, b.Y) {
				continue
			}
			if !friendlyFire && a.Faction == b.Faction {
				continue
			}
			nx, ny, ok := unitTo(a.X, a.Y, b.X, b.Y)
			if !ok {
				continue
			}
			aMax := a.Cfg.MaxSpeed * g.difficulty
			bMax := b.Cfg.MaxSpeed * g.difficulty
			if aMax > 0 && closingSpeed(a.VelX, a.VelY, nx, ny) >= impactThreshold(aMax) {
				applyImpactDamage(b)
			}
			if bMax > 0 && closingSpeed(b.VelX, b.VelY, -nx, -ny) >= impactThreshold(bMax) {
				applyImpactDamage(a)
			}
		}
	}
}

// checkCollisions — [БІЙ] тепер перевіряє не дотик, а СМЕРТЬ гравця (HP ≤ 0).
// Саму шкоду завдає resolveImpacts.
func (g *Game) checkCollisions() {
	if g.player.HP > 0 {
		return
	}
	g.metrics.catches++ // [МЕТРИКИ] вбивство гравця (для catch-rate)
	if aiPlayer {
		// [SELF-PLAY] Не game over — переносимо жертву й тренуємось далі.
		g.respawnPlayer()
	} else {
		g.gameOver = true
		g.saveBrains() // зберігаємо мозок хижаків
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
