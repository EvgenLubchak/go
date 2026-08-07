package main

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// updateBoidMap очищає сітку і розставляє ворогів.
func (g *Game) updateBoidMap() {
	for y := range g.boidMap {
		for x := range g.boidMap[y] {
			g.boidMap[y][x] = 0
		}
	}
	for i, e := range g.units {
		cx := int(e.X) / pixelSize
		cy := int(e.Y) / pixelSize
		if cx >= 0 && cx < boidMapW && cy >= 0 && cy < boidMapH {
			g.boidMap[cy][cx] = i + 1
		}
	}
}

// calcAcceleration рахує alignment (boids) і chase для кожного ворога.
//
// [GO: GOROUTINES + SYNC.WAITGROUP]
// Розбиваємо ворогів на chunks і обраховуємо кожен у окремому goroutine.
// WaitGroup лічить активні goroutines: Add(1) перед запуском, Done() всередині,
// Wait() блокує поки всі не завершились.
//
// Worker pool: runtime.NumCPU() goroutines замість одного на кожного ворога —
// мінімальний overhead при максимальному паралелізмі.
func (g *Game) calcAcceleration() {
	n := len(g.units)
	if n == 0 {
		return
	}

	// [GO: SNAPSHOT PATTERN]
	// Копіюємо VelX/VelY/X/Y всіх ворогів перед паралельним обрахунком.
	// Goroutines читають snapshot (незмінний) → пишуть тільки у свій AccX/AccY.
	// X/Y потрібні для cohesion: середня позиція сусідів (центр маси).
	type snap struct {
		VelX, VelY, X, Y float32
		Faction          int // [КОМАНДИ] щоб флокуватись лише зі СВОЇМИ
	}
	snaps := make([]snap, n)
	for i := range g.units {
		u := &g.units[i]
		snaps[i] = snap{u.VelX, u.VelY, u.X, u.Y, u.Faction}
	}

	// Ділимо ворогів рівномірно між CPU ядрами
	numWorkers := runtime.NumCPU()
	chunkSize := (n + numWorkers - 1) / numWorkers // округлення вгору

	// [GO: SYNC.WAITGROUP]
	// var wg sync.WaitGroup — лічильник goroutines.
	// wg.Add(1) перед go func → wg.Done() при завершенні → wg.Wait() чекає всіх.
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}

		wg.Add(1)

		// [GO: GO FUNC з параметрами]
		// start і end передаємо як аргументи — інакше замикання захопить змінну
		// по посиланню і всі goroutines побачать одне й те саме значення на момент запуску.
		go func(start, end int) {
			defer wg.Done() // [GO: DEFER] — гарантовано викличеться при виході з функції

			for i := start; i < end; i++ {
				e := &g.units[i]
				cx := int(e.X) / pixelSize
				cy := int(e.Y) / pixelSize

				var avgVX, avgVY float32
				var avgX, avgY float32 // cohesion: центр маси сусідів
				var sepX, sepY float32 // separation: сума векторів відштовхування
				flockCount := 0        // [КОМАНДИ] лише СВОЇ — для alignment/cohesion

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
						nb := &snaps[idx-1]

						// [GO: SEPARATION] — з УСІМА сусідами, хоч би якої фракції:
						// юніти фізично не мають накладатись, і свої, і чужі.
						// Вектор від сусіда до мене; ділимо на відстань, тож ближчий
						// сусід відштовхує сильніше.
						rdx := e.X - nb.X
						rdy := e.Y - nb.Y
						d := float32(math.Sqrt(float64(rdx*rdx + rdy*rdy)))
						if d > 0 {
							sepX += rdx / d
							sepY += rdy / d
						}

						// [КОМАНДИ] Alignment/cohesion — ЛИШЕ зі своїми. Інакше дві
						// команди «вирівнювались» би одна з одною замість того, щоб
						// битись. (Зараз обидва коефіцієнти 0, тож фільтр — на майбутнє,
						// коли захочемо ввімкнути флокінг у якогось типу.)
						if nb.Faction != e.Faction {
							continue
						}
						avgVX += nb.VelX
						avgVY += nb.VelY
						avgX += nb.X
						avgY += nb.Y
						flockCount++
					}
				}

				e.AccX, e.AccY = 0, 0
				if flockCount > 0 {
					fc := float32(flockCount)

					// Alignment: тягнемо швидкість до середньої швидкості СВОЇХ
					e.AccX += (avgVX/fc - e.VelX) * e.Cfg.AlignmentRate
					e.AccY += (avgVY/fc - e.VelY) * e.Cfg.AlignmentRate

					// Cohesion: тягнемо до центру маси СВОЇХ
					e.AccX += (avgX/fc - e.X) * e.Cfg.CohesionRate
					e.AccY += (avgY/fc - e.Y) * e.Cfg.CohesionRate
				}
				// Separation: сума відштовхувань від УСІХ сусідів (не ділимо на
				// кількість — це сума, а не середнє).
				e.AccX += sepX * e.Cfg.SeparationRate
				e.AccY += sepY * e.Cfg.SeparationRate

				// [КОМАНДИ] ЦІЛЬ ЗАЛЕЖИТЬ ВІД ФРАКЦІЇ:
				//   рій і вбивці      → полюють на ГРАВЦЯ (їхні ваги навчені саме
				//                       під це, тож ми їх не інвалідуємо);
				//   юніти гравця      → на найближчого ВОРОГА (перехоплення).
				// Відповідь ворога виникає сама: resolveImpacts симетричний, тож
				// коли рій налітає на твого юніта — шкоду отримують обидва.
				//
				// [GO: БЕЗПЕКА] Позиції в цій фазі ніхто не пише (пишемо лише свій
				// AccX/AccY), тож читати g.units із горутин безпечно.
				target := g.nearestTargetFor(e)
				if target == nil {
					continue // супротивників не лишилось — полювати нема на кого
				}
				fdx := target.X - e.X
				fdy := target.Y - e.Y
				dist := float32(math.Sqrt(float64(fdx*fdx + fdy*fdy)))

				if e.Brain != nil {
					// [RL: Q-LEARNING — агент без вчителя]
					// 1. state: куди гравець + 8 whiskers (зір на стіни)
					// 2. Step: оцінює минулу дію за reward і обирає нову (ε-greedy)
					// 3. дія = один з 8 напрямків → прискорення туди
					// e.HitWall (наслідок минулого руху, виставлений у updateUnits)
					// стає сигналом штрафу за зіткнення зі стіною.
					// [ВБИВЦЯ] Той самий Step, але ІНШИЙ набір входів: замість
					// прямого напрямку — flow-field (шлях крізь стіни).
					// [КОМАНДИ] Поле беремо ЗА ФРАКЦІЄЮ: кожна сторона читає те,
					// що веде до супротивника (flowFor). Тому вбивці працюють
					// симетрично — і ворожі, і твої.
					var state [baseInputs]float32
					if e.Cfg.UsesFlowField {
						state = GatherKillerInputs(e, target, g.flowFor(e))
					} else {
						state = GatherInputs(e, target)
					}
					action := e.Brain.Step(state, e.HitWall)

					// [УХИЛЕННЯ] Девʼята дія не має напрямку: вона дає вікно
					// невразливості, а не прискорення. І поки воно триває, юніт НЕ
					// прискорюється зовсім — це і є ціна дії.
					//
					// Без ціни «тиснути щойно перезарядилось» було б слабко домінантною
					// СТАЛОЮ політикою: агент отримав би 10% пом'якшення, нічого не
					// вивчивши. З ціною виникає справжній компроміс — ухилення коштує
					// втраченої швидкості зближення, тобто втраченого удару.
					if action == actionDodge {
						if e.DodgeCooldown == 0 {
							e.DodgeTimer = dodgeInvuln
							e.DodgeCooldown = dodgeCooldown
							dodgeBurst(e, target)
						}
						// Перезарядка ще йде — дію змарновано. Це навмисно: інакше
						// спам був би безкарний.
					} else if e.DodgeTimer == 0 {
						e.AccX += dirs8[action][0] * brainForce * g.difficulty
						e.AccY += dirs8[action][1] * brainForce * g.difficulty
					}

					// [СТИГМЕРГІЯ] Відштовхування від слідів фрустрації навколо (лише
					// якщо феромони ввімкнені): рій уникає місць, де вже застрягав.
					if pheromonesEnabled {
						ffx, ffy := g.frustrationForce(e.X, e.Y)
						e.AccX += ffx
						e.AccY += ffy
					}
				} else if dist > 0 && e.Aggression > 0 {
					// Звичайні вороги: hardcoded chase з pounce
					if dist < e.Cfg.DetectionRange {
						pounce := (1 - dist/e.Cfg.DetectionRange) * e.Cfg.PounceMulti
						e.AccX += (fdx / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
						e.AccY += (fdy / dist) * e.Cfg.AggressionForce * e.Aggression * g.difficulty * (1 + pounce)
					}
				}
			}
		}(start, end)
	}

	// [GO: WAWG.WAIT]
	// Блокуємо головний goroutine поки всі workers не завершать свій chunk.
	// Тільки після цього updateUnits() отримає актуальні AX/AY.
	wg.Wait()
}

// updateUnits застосовує блукання, burst, прискорення, damping, рух і відбивання.
func (g *Game) updateUnits() {
	g.decayFrustration() // [СТИГМЕРГІЯ] сліди тануть щокадру (однопотоково)

	// [УХИЛЕННЯ] Таймери дії. Окремим циклом до фізики: DodgeTimer читає
	// applyImpactDamage, і він мусить бачити стан ЦЬОГО кадру.
	for i := range g.units {
		if g.units[i].DodgeTimer > 0 {
			g.units[i].DodgeTimer--
		}
		if g.units[i].DodgeCooldown > 0 {
			g.units[i].DodgeCooldown--
		}
	}

	for i := range g.units {
		e := &g.units[i]

		// [RL] Скидаємо прапор удару — фіксуємо зіткнення саме цього кадру.
		// calcAcceleration наступного кадру прочитає його як сигнал штрафу.
		e.HitWall = false

		// [GO: e.Cfg.WanderStrength] — Speeder блукає хаотично, Predator — плавно
		e.VelX += (rand.Float32() - 0.5) * e.Cfg.WanderStrength
		e.VelY += (rand.Float32() - 0.5) * e.Cfg.WanderStrength

		// Burst: Speeder б'є часто і сильно, Boid — рідко і слабко
		if rand.Float32() < e.Cfg.BurstChance {
			angle := rand.Float64() * 2 * math.Pi
			e.VelX += float32(math.Cos(angle)) * e.Cfg.BurstForce
			e.VelY += float32(math.Sin(angle)) * e.Cfg.BurstForce
		}

		e.VelX += e.AccX
		e.VelY += e.AccY

		e.VelX *= damping
		e.VelY *= damping

		// [GO: e.Cfg.MaxSpeed] — стеля швидкості своя у кожного типу.
		// [БІЙ] Поки триває відліт після удару, стеля піднята: інакше кліп зʼїв би
		// віддачу за перший же кадр і «кидок кобри» лишився б мікрорухом.
		currentMaxSpeed := e.Cfg.MaxSpeed * g.difficulty
		if e.KnockTimer > 0 {
			e.KnockTimer--
			currentMaxSpeed *= knockSpeedMulti
		}
		// [ВІДКИД] Поки триває ухилення, стеля піднята — інакше кліп зʼїв би стрибок за
		// один кадр, точно як він зʼїдав би віддачу удару. Та сама механіка, інша
		// причина, тож окремий множник, а не спільний із knockSpeedMulti.
		if e.DodgeTimer > 0 && dodgeDashSpeed > currentMaxSpeed {
			currentMaxSpeed = dodgeDashSpeed
		}
		speed := float32(math.Sqrt(float64(e.VelX*e.VelX + e.VelY*e.VelY)))
		if speed > currentMaxSpeed {
			e.VelX = e.VelX / speed * currentMaxSpeed
			e.VelY = e.VelY / speed * currentMaxSpeed
		}

		if e.HitTimer > 0 {
			e.HitTimer--
		}
		if e.InvulnTimer > 0 { // [БІЙ] кадри невразливості після удару
			e.InvulnTimer--
		}

		// Рух: відбивання від тайлових стін і країв екрану.
		newX := e.X + e.VelX
		newY := e.Y + e.VelY

		if !isWallRect(newX, e.Y) {
			e.X = newX
		} else {
			e.VelX = -e.VelX
			e.HitWall = true
		}
		if !isWallRect(e.X, newY) {
			e.Y = newY
		} else {
			e.VelY = -e.VelY
			e.HitWall = true
		}

		// Додатковий захист від виходу за межі (якщо ворог якось вийшов)
		if e.X < 0 {
			e.X = 0
			e.VelX = -e.VelX
			e.HitWall = true
		}
		if e.X > screenWidth-pixelSize {
			e.X = screenWidth - pixelSize
			e.VelX = -e.VelX
			e.HitWall = true
		}
		if e.Y < 0 {
			e.Y = 0
			e.VelY = -e.VelY
			e.HitWall = true
		}
		if e.Y > screenHeight-pixelSize {
			e.Y = screenHeight - pixelSize
			e.VelY = -e.VelY
			e.HitWall = true
		}

		updateFur(e)
		updateBalls(e)
		updateTentacle(e) // [ВОРС] щупальця тягнуться за юнітом (і показують його стан)
		updateBody(e)     // [ТІЛО] пружина розміру → пульс на зупинці

		// [СТИГМЕРГІЯ] Учень ОФІЦІЙНО застряг (спрацювала фрустрація) → лишаємо слід
		// саме в цій клітинці (лише якщо феромони ввімкнені). НЕ на кожен дотик
		// стіни (інакше «слимачий слід»), а лише в реальних глухих кутах.
		if pheromonesEnabled && e.Brain != nil && e.Brain.markStuck {
			e.Brain.markStuck = false
			cx := int(e.X) / pixelSize
			cy := int(e.Y) / pixelSize
			if cx >= 0 && cx < boidMapW && cy >= 0 && cy < boidMapH {
				g.frustration[cy][cx] += frustrationDeposit
			}
		}
	}
}

// decayFrustration притлумлює всі сліди феромонів (однопотоково, щокадру).
// Завдяки затуханню «погане місце» з часом забувається й стає прохідним знову.
func (g *Game) decayFrustration() {
	for y := range g.frustration {
		for x := range g.frustration[y] {
			g.frustration[y][x] *= frustrationDecay
		}
	}
}

// frustrationForce — сила відштовхування від слідів феромонів навколо точки.
// Сумуємо вектори «від клітинки-сліду до агента», зважені силою сліду й поділені
// на відстань (ближчий слід штовхає сильніше). Лише ЧИТАННЯ сітки → безпечно
// в паралельному calcAcceleration (запис відбувається в окремій фазі).
func (g *Game) frustrationForce(x, y float32) (fx, fy float32) {
	cx := int(x) / pixelSize
	cy := int(y) / pixelSize
	for dy := -frustrationRadius; dy <= frustrationRadius; dy++ {
		for dx := -frustrationRadius; dx <= frustrationRadius; dx++ {
			nx, ny := cx+dx, cy+dy
			if nx < 0 || nx >= boidMapW || ny < 0 || ny >= boidMapH {
				continue
			}
			f := g.frustration[ny][nx]
			if f <= 0 {
				continue
			}
			rx := x - (float32(nx)*pixelSize + pixelSize/2)
			ry := y - (float32(ny)*pixelSize + pixelSize/2)
			d := float32(math.Sqrt(float64(rx*rx + ry*ry)))
			if d < 1 {
				d = 1
			}
			fx += rx / d * f
			fy += ry / d * f
		}
	}
	return fx * frustrationRepel, fy * frustrationRepel
}

// nearestTargetFor — [КОМАНДИ] найближчий супротивник для юніта.
//
// Чому окремий метод, а не просто nearestHostile: ГРАВЕЦЬ живе не в g.units, а
// окремим полем Game. Для ворогів він теж легітимна ціль (і зазвичай головна),
// тож перебираємо і юнітів чужої фракції, І гравця, якщо він з іншого боку.
func (g *Game) nearestTargetFor(u *Pixel) *Pixel {
	best := nearestHostile(u, g.units)
	if u.Faction == g.player.Faction || g.player.HP <= 0 {
		return best
	}
	pdx, pdy := g.player.X-u.X, g.player.Y-u.Y
	pd := pdx*pdx + pdy*pdy
	if best == nil {
		return &g.player
	}
	bdx, bdy := best.X-u.X, best.Y-u.Y
	if pd < bdx*bdx+bdy*bdy {
		return &g.player
	}
	return best
}

// updateBalls — [КУЛЬКИ] один крок фізики. Той самий принцип, що у ворсі: кулька
// підтягується до своєї точки лише на частку шляху й принципово не встигає.
//
// Саме з відставання виходить уся анімація: рушив — кульки лишились позаду й
// підтягуються; різко повернув — заносить убік; спинився — доганяють і завмирають.
// Нічого з цього не програмується окремо.
func updateBalls(u *Pixel) {
	cx := u.X + pixelSize/2
	cy := u.Y + pixelSize/2
	for i := 0; i < ballCount; i++ {
		hx, hy := ballHome(i, cx, cy)
		u.Balls[i][0] += (hx - u.Balls[i][0]) * ballStiff
		u.Balls[i][1] += (hy - u.Balls[i][1]) * ballStiff
	}
}

// updateTentacle — [ВІДРОСТОК] один крок фізики щупальця.
//
// Кожен суглоб тягнеться за ПОПЕРЕДНІМ (а не за тілом), і жорсткість падає вниз по
// ланцюжку. Саме з цих двох правил виходить хвиля, що біжить від основи до кінчика, і
// S-подібний вигин на поворотах.
//
// Ціль суглоба — «попередній плюс сегмент УНИЗ». Не «продовжити напрямок попереднього»,
// як у ворсі: ворс стирчить навсібіч і має лишатись прямим у спокої, а щупальце мусить
// ОБВИСАТИ. З «уніз» у спокої воно висить рівно, а на русі відставання саме вигинає його.
func updateTentacle(u *Pixel) {
	cx := u.X + pixelSize/2
	cy := u.Y + pixelSize/2
	px, py := tentRoot(cx, cy)
	stiff := float32(tentStiff)
	for i := 0; i < tentJoints; i++ {
		u.Tent[i][0] += (px - u.Tent[i][0]) * stiff
		u.Tent[i][1] += (py + tentSeg - u.Tent[i][1]) * stiff
		px, py = u.Tent[i][0], u.Tent[i][1]
		stiff *= tentFalloff
	}
}

// dodgeBurst — [ВІДКИД] стрибок убік від лінії загрози.
//
// ПЕРПЕНДИКУЛЯРНО, а не «геть»: ривок гравця криє 50px, тож відхід назад його не
// рятує — він просто дожене. Убік вистачає корпуса.
//
// НАПРЯМОК ОБИРАЄ ФІЗИКА, а не мережа, і це принципово. Якби бік вибирала мережа, у
// задачу повернувся б ДОБУТОК (дія = f(телеграф, геометрія)) — а ми щойно виміряли, що
// саме добуток із затримкою її й ламає (драбина: 0 з 4 проти 95.6% без затримки).
// Рішення лишається суто ЧАСОВИМ — тим режимом, який учень бере.
//
// Агент при цьому не позбавлений впливу: бік визначається тим, куди юніт УЖЕ хилиться,
// тобто його власним позиціюванням до моменту ухилення.
func dodgeBurst(e *Pixel, threat *Pixel) {
	if threat == nil {
		return
	}
	tx, ty := e.X-threat.X, e.Y-threat.Y
	d := float32(math.Sqrt(float64(tx*tx + ty*ty)))
	if d < 0.001 {
		return // збіглись у точку — перпендикуляра немає
	}
	px, py := -ty/d, tx/d // поворот на 90°
	if e.VelX*px+e.VelY*py < 0 {
		px, py = -px, -py // у той бік, куди вже рухався
	}
	e.VelX, e.VelY = px*dodgeDashSpeed, py*dodgeDashSpeed
}

// updateFur — [ВОРС] один крок фізики хутра.
//
// Уся анімація тримається на ОДНІЙ ідеї — ВІДСТАВАННІ: кожен суглоб щокадру
// підтягується до своєї «бажаної» позиції лише на частку шляху (stiff). Він
// принципово не встигає, і з цього само собою виходить:
//
//	рух уперед    → ворс лягає назад (кінчики відстали)
//	різкий поворот→ ворс хльоскає
//	зупинка       → плавно осідає
//	більша швидкість → сильніше притиснутий
//
// Жодне з цих правил не написане окремо — вони наслідок лагу.
//
// [ПРИЛАД] Ворс заразом показує внутрішній стан мозку:
//
//	фрустрація (застряг) → НАЇЖАЧУЄТЬСЯ: довший ворс + висока жорсткість
//	                       (майже без лагу → стирчить радіально, як у кота)
//	низьке HP            → ОБВИСАЄ: коротший і млявіший
//	розгін на удар       → сам собою лягає назад (нічого не додаємо)
//
// updateBody — пружина розміру тіла. Джерело ПУЛЬСУ.
//
// Ціль розміру падає зі швидкістю: медуза стискається, коли жене крізь воду.
// Поточний розмір наздоганяє ціль через пружину з НЕДОСТАТНІМ затуханням — тому,
// коли юніт різко спиняється, ціль стрибає вгору, тіло проскакує її і кілька разів
// закачується. Пульс ніде не написаний, він наслідок перельоту — той самий принцип,
// що й розвівання щупалець, тож обидві анімації синхронізуються самі собою.
// Намальована синусоїда билася б із ними двома незалежними ритмами.
func updateBody(u *Pixel) {
	maxSpd := u.Cfg.MaxSpeed
	if maxSpd <= 0 {
		maxSpd = playerBaseSpeed // гравець живе поза unitRoster, у нього Cfg порожній
	}
	spd := float32(math.Sqrt(float64(u.VelX*u.VelX + u.VelY*u.VelY)))
	frac := spd / maxSpd
	if frac > 1 {
		frac = 1
	}
	target := 1 - bodySquash*frac

	u.BodyVel += (target - u.BodyScale) * bodyStiff
	u.BodyVel *= bodyDamp
	u.BodyScale += u.BodyVel
}

func updateFur(u *Pixel) {
	cx := u.X + pixelSize/2
	cy := u.Y + pixelSize/2

	length := float32(furLen)
	stiff := float32(furStiffness)

	if u.Brain != nil && u.Brain.frustration > 0 {
		length *= furBristleLen
		stiff = furBristleStiff
	}
	if u.MaxHP > 0 { // поранений — ворс коротшає й стає млявішим
		hp := float32(u.HP) / float32(u.MaxHP)
		k := furHurtLen + (1-furHurtLen)*hp
		length *= k
		stiff *= k
	}

	half := length * 0.5
	for i := 0; i < furStrands; i++ {
		dx, dy := dirs8[i][0], dirs8[i][1]

		// Суглоб 1 (середина) тягнеться до точки на пів-довжини від центру.
		mx := &u.Fur[i][0][0]
		my := &u.Fur[i][0][1]
		*mx += (cx + dx*half - *mx) * stiff
		*my += (cy + dy*half - *my) * stiff

		// Суглоб 2 (кінчик) тягнеться за СЕРЕДИНОЮ, продовжуючи її напрямок від
		// центру — саме тому ворсинка ВИГИНАЄТЬСЯ, а не лишається прямою палицею.
		ex, ey := *mx-cx, *my-cy
		if l := float32(math.Sqrt(float64(ex*ex + ey*ey))); l > 0.001 {
			ex, ey = ex/l, ey/l
		} else {
			ex, ey = dx, dy
		}
		tx := &u.Fur[i][1][0]
		ty := &u.Fur[i][1][1]
		*tx += (*mx + ex*half - *tx) * stiff
		*ty += (*my + ey*half - *ty) * stiff
	}
}
