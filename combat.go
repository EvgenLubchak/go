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
	// [УХИЛЕННЯ] Активне ухилення захищає так само, як невразливість після удару.
	// Окремим таймером, а не через InvulnTimer: у них різна тривалість і різний сенс,
	// і змішувати їх означало б, що вдалий ухил дає ще й 45 кадрів безкарності.
	if target.InvulnTimer > 0 || target.DodgeTimer > 0 {
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
			// [ВІДСІЧ] Персональна, якщо тип її задав. Береться в НАПАДНИКА — це його
			// власна віддача від удару, а не властивість цілі.
			recoil := float32(knockbackRecoil)
			if attacker.KnockRecoil > 0 {
				recoil = attacker.KnockRecoil
			}
			aImp := knockbackImpulse * recoil * (1 - attacker.KnockResist)

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
		// Вікно для віддачі дає САМА МЕХАНІКА, а не невразливість: поза активною фазою
		// гравець не ранить нічим, тож усі dashRecovery кадрів агент може бити безкарно
		// незалежно від свого InvulnTimer. Я спершу приписав це збігу двох таймерів —
		// невірно, збіг був випадковий і зник, коли замах виріс до 60.
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

// [СТИЛЬ ПРИЦІЛУ] Два способи, якими нападник обирає ЛІНІЮ удару. Різниця не
// косметична: контрзаходи до них ВЗАЄМОВИКЛЮЧНІ, і саме це робить задачу про
// передбачення непорожньою.
//
//	ВЕДУЧИЙ      цілиться туди, куди ціль ЛЕТИТЬ  → рятує РОЗВОРОТ
//	ДЗЕРКАЛЬНИЙ  передбачає розворот і цілиться ТУДИ → рятує НЕ розвертатись
//
// Те, що рятує від одного, гарантовано ловить від іншого. Ставки однакові (−dashDamage
// за хибний вибір, 0 за правильний), тож сліпий агент приречений на 50/50, а той, хто
// знає стиль, — на ~100%. Виграш ~4 одиниці шкоди за цикл.
//
// Це замінило конструкцію «ранній/пізній замах», яку ми відкинули АРИФМЕТИКОЮ ще до
// коду: там «завжди відступати» домінувало проти обох стилів (виграш оракула +0.0 при
// реалістичних частках ухилення), бо ривок бʼє 8, а таран 1 — жодна кількість таранів
// не окупає зʼїденого ривка.
var dashStyleMirror bool

// dashLeadFrames — на скільки кадрів уперед екстраполюємо рух цілі.
//
// Замах (60) + ривок (5) = 65 кадрів до влучання, але беремо 40: при швидкості
// стражника 0.6 це 24px ≈ один корпус — досить, щоб промах був справжнім, і водночас
// точка прицілу лишається в межах досяжності ривка (50px).
const dashLeadFrames = 40

// dashAimAt — напрямок ривка по цілі згідно з поточним стилем. Ненормований.
func dashAimAt(from, target *Pixel) (float32, float32) {
	lead := float32(dashLeadFrames)
	if dashStyleMirror {
		lead = -lead // цілимось туди, куди ціль піде, ЯКЩО РОЗВЕРНЕТЬСЯ
	}
	return target.X + target.VelX*lead - from.X, target.Y + target.VelY*lead - from.Y
}

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

// dashRecoveryLeft — [ВХІД МОЗКУ] скільки ще триває ВІКНО ПОКАРАННЯ: 1 на початку
// відходу, 0 у кінці й поза ним.
//
// ⚠️ ЦЯ ОЗНАКА ЗАЙНЯЛА СЛОТ, ЯКИЙ Я ПЛАНУВАВ ДЛЯ ВЛАСНОЇ НЕВРАЗЛИВОСТІ, і причина
// варта запису. Я доводив, що агент мусить бачити свій InvulnTimer, бо після удару
// має 45 кадрів безкарної відповіді. Це було правдою, ПОКИ була ульта: вона била
// щодесять кадрів, і невразливість справді була вікном.
//
// Ривок цю цінність прибрав. Гравець тепер ранить ЛИШЕ в активній фазі, тож поза нею
// агент безкарний БЕЗ жодної невразливості — вікно дає механіка, а не таймер. Ознака
// стала майже пустою саме в тому сценарії, який ми міряємо.
//
// Тому слот віддано протилежній половині задачі: замах (слот 15) каже, коли
// УХИЛЯТИСЬ, відхід (слот 14) — коли КАРАТИ. Обидві половини подані явно, і це
// свідомо: спершу стеля з явними ознаками, потім прибираємо їх і дивимось, чи вікно
// памʼяті їх замінить.
//
// Невразливість лишається цікавою для ПОВНОЇ гри, де стражника таранять ще й союзні
// юніти. У заміру з одним гравцем вона ні на що не впливає.
func dashRecoveryLeft(p *Pixel) float32 {
	if p.DashPhase != dashPhaseRecovery {
		return 0
	}
	return float32(p.DashTimer) / float32(dashRecovery)
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
