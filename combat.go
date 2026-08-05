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

// pushOffPlayer — юніт не може стояти ВСЕРЕДИНІ гравця.
//
// [ЧОМУ ЦЕ ПОТРІБНО] Separation рахується з boidMap (updateBoidMap), а туди
// потрапляють ЛИШЕ g.units — гравця там немає. Тобто юніти ніколи не відштовхувались
// від гравця й могли стояти буквально в ньому. Уп'яте вилізла та сама асиметрія
// «гравець живе поза g.units» (до цього: ворс, нагорода, вибір цілі, метрики).
//
// Поки не було тертя, дірка не проявлялась: юніт пролітав повз і контакт був
// миттєвим. З damping=0.95 він навчився паркуватись — і почав завдавати шкоди
// ОБІЙМАМИ. Механіка ж задумувалась протилежною: «повільно зіштовхнулись = нічого;
// налетів = вкусив». А юніт усередині гравця продовжує прискорюватись на brainForce
// щокадру, тримає швидкість 1.2 при порозі 0.72 — і формально «налітає на повній»
// постійно. Смерть за ~4 секунди обіймів.
//
// Лікування — те саме, що для стін в updateUnits: юніт ВІДБИВАЄТЬСЯ. Спершу я тут
// швидкість гасив, і вийшло гірше за початкову ваду: юніт застрягав на поверхні
// назавжди, бо за кадр мозок додає лише brainForce 0.3, а поріг удару 0.72 (у вбивці
// 0.96) — розігнатись повторно він не встигав ніколи, і удари зникли зовсім.
//
// Відбивання дає задуманий ритм: налетів → вкусив → відскочив → заходить знову.
// А коли юніт просто тиснеться, він відскакує зі своєю ж мізерною швидкістю й
// порогу не досягає — обійми так само не кусають.
//
// Виштовхуємо ЛИШЕ юніта, не гравця: інакше рій зміг би затовкти тебе крізь стіну.
// Якщо виштовхувати нікуди (там стіна) — позицію лишаємо, але швидкість гасимо
// однаково: саме вона визначає шкоду.
//
// Діє на юнітів БУДЬ-ЯКОЇ фракції — це фізика тіл, а не бойове правило.
// nudgeX/nudgeY — зсув тіла з перевіркою стіни. Якщо там стіна, зсув не робимо:
// краще лишити перекриття на кадр, ніж заштовхнути юніта в геометрію.
func nudgeX(u *Pixel, d float32) {
	if !isWallRect(u.X+d, u.Y) {
		u.X += d
	}
}

func nudgeY(u *Pixel, d float32) {
	if !isWallRect(u.X, u.Y+d) {
		u.Y += d
	}
}

// separateUnits — те саме, що pushOffPlayer, але для ДВОХ РУХОМИХ тіл.
//
// Різниця з гравцем одна: маси рівні, тож розходяться обидва — кожен на половину
// перекриття, і кожен відбивається своєю складовою швидкості. Гравця ж ми не рухаємо
// зовсім, інакше рій міг би затовкти його крізь стіну.
//
// Діє МІЖ УСІМА юнітами, незалежно від фракції: це фізика тіл, а не бойове правило.
// Наслідок, який варто знати: щільна купа більше неможлива — рій утворює передній
// ряд, і задні фізично не дістають до цілі, поки передні її не звільнять.
func separateUnits(a, b *Pixel) {
	dx := a.X - b.X
	dy := a.Y - b.Y
	ox := pixelSize - float32(math.Abs(float64(dx)))
	oy := pixelSize - float32(math.Abs(float64(dy)))
	if ox <= 0 || oy <= 0 {
		return
	}

	if ox < oy { // вісь меншого занурення
		half := ox / 2
		if dx >= 0 {
			nudgeX(a, half)
			nudgeX(b, -half)
			if a.VelX < 0 {
				a.VelX *= -bodyBounce
			}
			if b.VelX > 0 {
				b.VelX *= -bodyBounce
			}
			return
		}
		nudgeX(a, -half)
		nudgeX(b, half)
		if a.VelX > 0 {
			a.VelX *= -bodyBounce
		}
		if b.VelX < 0 {
			b.VelX *= -bodyBounce
		}
		return
	}

	half := oy / 2
	if dy >= 0 {
		nudgeY(a, half)
		nudgeY(b, -half)
		if a.VelY < 0 {
			a.VelY *= -bodyBounce
		}
		if b.VelY > 0 {
			b.VelY *= -bodyBounce
		}
		return
	}
	nudgeY(a, -half)
	nudgeY(b, half)
	if a.VelY > 0 {
		a.VelY *= -bodyBounce
	}
	if b.VelY < 0 {
		b.VelY *= -bodyBounce
	}
}

func (g *Game) pushOffPlayer(u *Pixel) {
	if g.player.HP <= 0 || !collides(g.player.X, g.player.Y, u.X, u.Y) {
		return
	}
	dx := u.X - g.player.X
	dy := u.Y - g.player.Y
	ox := pixelSize - float32(math.Abs(float64(dx))) // глибина перекриття по X
	oy := pixelSize - float32(math.Abs(float64(dy)))
	if ox <= 0 || oy <= 0 {
		return
	}

	// Виштовхуємо по осі МЕНШОГО занурення — так юніт ковзає вздовж грані гравця,
	// а не телепортується через нього.
	if ox < oy {
		if dx >= 0 {
			nudgeX(u, ox)
			if u.VelX < 0 {
				u.VelX *= -bodyBounce
			}
		} else {
			nudgeX(u, -ox)
			if u.VelX > 0 {
				u.VelX *= -bodyBounce
			}
		}
		return
	}
	if dy >= 0 {
		nudgeY(u, oy)
		if u.VelY < 0 {
			u.VelY *= -bodyBounce
		}
	} else {
		nudgeY(u, -oy)
		if u.VelY > 0 {
			u.VelY *= -bodyBounce
		}
	}
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
//
// [БІЙ: АТРИБУЦІЯ] Записує подію обом сторонам у лічильники мозку — це
// сировина для бойової нагороди (rewardFor споживає й обнуляє їх наступного
// кадру). attacker може бути nil (напр. шкода не від агента).
func applyImpactDamage(attacker, target *Pixel) {
	if target.InvulnTimer > 0 {
		return
	}
	target.HP -= impactDamage
	target.HitTimer = hitFlashDuration // біле блимання (вже було для удару гравця)
	target.InvulnTimer = impactInvuln

	// [ВІДДАЧА] Ціль відлітає, нападник відсікається назад. Саме ТУТ, а не в
	// separateUnits: віддача належить УДАРУ, а не дотику. Дотик дає дрібний
	// пропорційний відскок (bodyBounce), удар — фіксований великий імпульс.
	if attacker != nil {
		if nx, ny, ok := unitTo(attacker.X, attacker.Y, target.X, target.Y); ok {
			// [ВАГА] KnockResist: 0 = відлітає повністю, 1 = не рухається зовсім.
			// Потрібен для «важких» типів: стражник, який відлітає на три корпуси від
			// кожного удару, перестає бути стражником — його задача тримати місце.
			tImp := knockbackImpulse * (1 - target.KnockResist)
			aImp := knockbackImpulse * knockbackRecoil * (1 - attacker.KnockResist)

			target.VelX += nx * tImp
			target.VelY += ny * tImp
			target.KnockTimer = knockFrames

			attacker.VelX -= nx * aImp
			attacker.VelY -= ny * aImp
			attacker.KnockTimer = knockFrames
		}
	}

	if attacker != nil && attacker.Brain != nil {
		attacker.Brain.dmgDealt += impactDamage
		if target.HP <= 0 {
			attacker.Brain.kills++ // добив — головна ціль бойової нагороди
		}
	}
	if target.Brain != nil {
		target.Brain.dmgTaken += impactDamage
	}
}

// resolveImpacts — [БІЙ] проходить пари, що перетинаються, і завдає шкоди за
// closing speed. Обробляє і гравець↔вороги, і вороги↔вороги.
// Однопотоково (після паралельної фази) → без гонок.
func (g *Game) resolveImpacts() {
	// Максимальна швидкість гравця — та сама формула, що в updatePlayer.
	playerMax := float32(playerBaseSpeed) * float32(math.Sqrt(float64(g.difficulty)))

	// --- Гравець ↔ вороги ---
	for i := range g.units {
		e := &g.units[i]
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
			applyImpactDamage(&g.player, e)
		}
		// Ворог кидається на гравця (напрямок навпаки).
		eMax := e.Cfg.MaxSpeed * g.difficulty
		if eMax > 0 && closingSpeed(e.VelX, e.VelY, -nx, -ny) >= impactThreshold(eMax) {
			applyImpactDamage(e, &g.player)
		}
	}

	// --- Вороги ↔ вороги (i<j, щоб кожну пару рахувати раз) ---
	for i := range g.units {
		a := &g.units[i]
		if a.HP <= 0 {
			continue
		}
		for j := i + 1; j < len(g.units); j++ {
			b := &g.units[j]
			if b.HP <= 0 || !collides(a.X, a.Y, b.X, b.Y) {
				continue
			}

			// ШКОДА — лише між ворожими (або за friendlyFire). Фізика нижче — для всіх.
			if friendlyFire || a.Faction != b.Faction {
				if nx, ny, ok := unitTo(a.X, a.Y, b.X, b.Y); ok {
					aMax := a.Cfg.MaxSpeed * g.difficulty
					bMax := b.Cfg.MaxSpeed * g.difficulty
					if aMax > 0 && closingSpeed(a.VelX, a.VelY, nx, ny) >= impactThreshold(aMax) {
						applyImpactDamage(a, b)
					}
					if bMax > 0 && closingSpeed(b.VelX, b.VelY, -nx, -ny) >= impactThreshold(bMax) {
						applyImpactDamage(b, a)
					}
				}
			}

			// ФІЗИКА — незалежно від фракції, і ОБОВʼЯЗКОВО після шкоди: інакше удар
			// рахувався б по вже відбитій швидкості (та сама пастка, що з гравцем).
			separateUnits(a, b)
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
	g.applyPlayerMelee()
}

// applyPlayerMelee — саме НАРАХУВАННЯ удару, окремо від читання клавіатури.
// Розділено, щоб це можна було перевірити тестом: ebiten у тесті недоступний.
func (g *Game) applyPlayerMelee() {
	px := g.player.X + pixelSize/2
	py := g.player.Y + pixelSize/2

	for i := range g.units {
		e := &g.units[i]
		ex := e.X + pixelSize/2
		ey := e.Y + pixelSize/2
		dx := px - ex
		dy := py - ey
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
		if dist <= attackRadius {
			e.HP -= attackDamage
			e.HitTimer = hitFlashDuration

			// [БІЙ: АТРИБУЦІЯ] Зараховуємо шкоду в мозок ЦІЛІ.
			//
			// Раніше цього не було, і це був справжній пропуск: удар пробілом міняв HP
			// напряму, обходячи applyImpactDamage, тож dmgTaken не реєструвався НІКОЛИ.
			// Отже всі агенти з CombatReward (вбивці, союзники) вчились не відчуваючи
			// ГОЛОВНОЇ зброї гравця — їхній rewardDamageTaken = −2 спрацьовував лише
			// від зіткнень. Для боса з ОДНОЮ бойовою нагородою це зробило б навчання
			// беззмістовним: половина подій, на які він мусить реагувати, була невидима.
			if e.Brain != nil {
				e.Brain.dmgTaken += attackDamage
				if e.HP <= 0 && g.player.Brain != nil {
					g.player.Brain.kills++ // [SELF-PLAY] жертві теж треба знати результат
				}
			}
			if g.player.Brain != nil {
				g.player.Brain.dmgDealt += attackDamage
			}
		}
	}
}

// deathTransition — [RL] доставляє агентові нагороду за ФАТАЛЬНИЙ удар.
//
// Без цього смерть була безкоштовною: юніта видаляли, наступного Step він не
// отримував, і dmgTaken від останнього удару ніколи не ставав −2. Тобто в нагороді не
// існувало причини не вмирати.
//
// Перехід позначаємо terminal: після смерті майбутнього немає, тож ціль Беллмана — це
// лише нагорода, без γ·maxQ(наступний стан).
//
// Лише СТЕК-шлях. У GRU нагорода живе у відрізках, і встромити термінальний крок
// посеред відрізка складніше: там частковий відрізок просто відкидається
// (resetForNewLife скидає seqN). Наразі всі наші учні на стеку, тож це покриває всіх;
// для GRU це лишається боргом.
func (g *Game) deathTransition(e *Pixel) {
	b := e.Brain
	if b == nil || b.net == nil || !b.hasPrev || b.net.mem.gru {
		return
	}
	reward := b.rewardFor(false, b.prevState[inWhisker0+b.prevAction])
	b.net.remember(transition{s: b.prevState, a: b.prevAction, r: reward, terminal: true})
}

// handleDeadUnits — [РЕСПАУН] оживляє тих, у кого лишились повернення, решту видаляє.
// [GO: FILTER SLICE in-place]
// g.units[:0] — той самий масив у пам'яті, але довжина 0.
// append пише поверх — без нової алокації пам'яті.
func (g *Game) handleDeadUnits() {
	alive := g.units[:0]
	for i := range g.units {
		e := &g.units[i]
		if e.HP > 0 {
			alive = append(alive, *e)
			continue
		}

		// Спершу ДОСТАВИТИ нагороду за смерть — до будь-якого скидання стану.
		g.deathTransition(e)

		if e.RespawnsLeft == 0 {
			continue // повернень немає → зникає назавжди
		}
		if e.RespawnsLeft > 0 {
			e.RespawnsLeft--
		} // негативне = безкінечно, не зменшуємо

		e.reviveAt(e.SpawnX, e.SpawnY)
		alive = append(alive, *e)
	}
	g.units = alive
}
