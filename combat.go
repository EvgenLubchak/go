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
func applyImpactDamage(attacker, target *Pixel, dmg int) {
	if target.InvulnTimer > 0 {
		return
	}
	target.HP -= dmg
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
		attacker.Brain.dmgDealt += dmg
		attacker.Brain.mDmgDealt += dmg // [МЕТРИКИ] не споживається rewardFor
		if target.HP <= 0 {
			attacker.Brain.kills++ // добив — головна ціль бойової нагороди
		}
	}
	if target.Brain != nil {
		target.Brain.dmgTaken += dmg
		target.Brain.mDmgTaken += dmg
	}
}

// resolveImpacts — [БІЙ] проходить пари, що перетинаються, і завдає шкоди за
// closing speed. Обробляє і гравець↔вороги, і вороги↔вороги.
// Однопотоково (після паралельної фази) → без гонок.
func (g *Game) resolveImpacts() {
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
		// [РИВОК] Гравець ранить ЛИШЕ в активній фазі. Перевірки швидкості більше
		// немає, і це не оптимізація, а зміна правила: раніше шкода була ПОБІЧНИМ
		// ЕФЕКТОМ того, що ти швидко їхав, тепер вона — РЕЗУЛЬТАТ рішення вдарити.
		//
		// Саме тут зникає домінантна стратегія «швидко кататись і таранити». Таран не
		// прибрано — його зроблено навмисним.
		//
		// Невразливість цілі після удару (impactInvuln = 45) майже точно накриває
		// відхід гравця (dashRecovery = 45): той, кого влучили, стає безкарним рівно
		// на той час, поки нападник безпорадний. Вікно для віддачі виникло саме,
		// з двох незалежно виведених чисел.
		if g.player.DashPhase == dashPhaseActive {
			applyImpactDamage(&g.player, e, dashDamage)
		}
		// Ворог кидається на гравця (напрямок навпаки).
		eMax := e.Cfg.MaxSpeed * g.difficulty
		if eMax > 0 && closingSpeed(e.VelX, e.VelY, -nx, -ny) >= impactThreshold(eMax) {
			applyImpactDamage(e, &g.player, impactDamage)
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
						applyImpactDamage(a, b, impactDamage)
					}
					if bMax > 0 && closingSpeed(b.VelX, b.VelY, -nx, -ny) >= impactThreshold(bMax) {
						applyImpactDamage(b, a, impactDamage)
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

// [РИВОК] Фази атаки гравця. Замінили ульту: та була AoE радіусом 120 раз на 10
// кадрів, тобто НЕВІДВОРОТНА, і бій зводився до молотіння. Тут атака має тривалість,
// а отже її можна прочитати й на неї можна відреагувати.
const (
	dashIdle          = iota // не атакує — керування вільне
	dashPhaseWindup          // замах: стоїть, напрямок замкнено, юніт це БАЧИТЬ
	dashPhaseActive          // ривок: летить по замкненій лінії, шкода на контакті
	dashPhaseRecovery        // відхід: керування заблоковане — вікно для покарання
)

// dashCycle — повна довжина атаки. Це знаменник в умові «ухилятись вигідніше за
// кемпінг» (див. dashDamage у main.go), тож живе константою, а не магічним числом.
const dashCycle = dashWindup + dashActive + dashRecovery

// startDash — почати атаку в напрямку (dx, dy). Окремо від читання клавіатури, щоб
// це можна було перевірити тестом: ebiten у тесті недоступний.
//
// Повертає false, якщо атака не почалась.
func startDash(p *Pixel, dx, dy float32) bool {
	if p.DashPhase != dashIdle {
		return false // [КОМІТ] почав — доводь. Скасувати замах не можна, у цьому й ціна.
	}
	n := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if n < 1e-6 {
		return false // без напрямку немає ЛІНІЇ удару, а отже й нічого читати
	}
	p.DashDirX, p.DashDirY = dx/n, dy/n
	p.DashPhase, p.DashTimer = dashPhaseWindup, dashWindup
	return true
}

// advanceDash — один кадр машини фаз.
func advanceDash(p *Pixel) {
	if p.DashPhase == dashIdle {
		return
	}
	p.DashTimer--
	if p.DashTimer > 0 {
		return
	}
	switch p.DashPhase {
	case dashPhaseWindup:
		p.DashPhase, p.DashTimer = dashPhaseActive, dashActive
	case dashPhaseActive:
		p.DashPhase, p.DashTimer = dashPhaseRecovery, dashRecovery
	default:
		p.DashPhase, p.DashTimer = dashIdle, 0
	}
}

// dashWindupProgress — [ВХІД МОЗКУ] наскільько замах уже визрів: 0..1, і 0 коли
// замаху немає. Це ЯВНА ознака телеграфу — свідомий контроль перед тим, як вимагати
// від агента вивести її з історії самому.
//
// Порядок «явне перед відкривним» — та дисципліна, якої нам забракло з GRU: там ми
// одразу вимагали ВИВЕСТИ памʼять і отримали нуль, не знаючи, чи задача взагалі
// розвʼязна. Тут спершу подаємо відповідь у вхід і встановлюємо СТЕЛЮ, а вже потім
// ознаку прибираємо й дивимось, чи вікно памʼяті її замінить.
func dashWindupProgress(p *Pixel) float32 {
	if p.DashPhase != dashPhaseWindup {
		return 0
	}
	return float32(dashWindup-p.DashTimer) / float32(dashWindup)
}

// playerDashInput — читає SPACE і напрямок, починає ривок. Напрямок беремо з
// НАТИСНУТИХ клавіш, а не з поточної швидкості: гравець мусить сам оголосити лінію
// удару, і в замаху вона вже не змінюється.
//
// [GO: inpututil.IsKeyJustPressed] — лише перший кадр натискання, інакше утримання
// пробілу перезапускало б замах щокадру.
func (g *Game) playerDashInput() {
	if !inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		return
	}
	var dx, dy float32
	if ebiten.IsKeyPressed(ebiten.KeyArrowUp) || ebiten.IsKeyPressed(ebiten.KeyW) {
		dy--
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowDown) || ebiten.IsKeyPressed(ebiten.KeyS) {
		dy++
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowLeft) || ebiten.IsKeyPressed(ebiten.KeyA) {
		dx--
	}
	if ebiten.IsKeyPressed(ebiten.KeyArrowRight) || ebiten.IsKeyPressed(ebiten.KeyD) {
		dx++
	}
	if dx == 0 && dy == 0 {
		// Клавіш не тримають — беремо напрямок руху. Стоячи на місці вдарити не можна:
		// без лінії удару немає ні атаки, ні телеграфу.
		dx, dy = g.player.VelX, g.player.VelY
	}
	startDash(&g.player, dx, dy)
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
