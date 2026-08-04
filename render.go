package main

import (
	"fmt"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	etext "github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// drawInputDirs малює, які напрямки руху зараз задіяні: чотири палички від центру,
// яскрава = задіяна, тьмяна = ні.
//
// ТЬМЯНІ МАЛЮЄМО НАВМИСНО. Якби показувались лише натиснуті, відсутність палички була
// б неоднозначною: «не тисну» чи «віджет узагалі не працює». З постійним хрестом
// зрозуміло завжди.
//
// Читаємо НЕ сиру клавіатуру, а обраний напрямок руху — тому в режимі aiPlayer той
// самий віджет показує, що робить мозок-жертва. Це прилад того ж роду, що форма тіла,
// а не дзеркало клавіш.
//
// У виді від першої особи клавіші означають інше (вліво/вправо — поворот камери,
// вгору/вниз — хід уперед/назад), і віджет це чесно відображає: він показує НАТИСНУТЕ,
// а не «куди полетить».
func (g *Game) drawInputDirs(screen *ebiten.Image) {
	// up, down, left, right
	var on [4]bool

	if aiPlayer && g.player.Brain != nil {
		// [SELF-PLAY] Напрямок з дії мережі: розкладаємо вектор dirs8 на осі.
		d := dirs8[g.player.Brain.lastAction]
		on[0] = d[1] < -0.01
		on[1] = d[1] > 0.01
		on[2] = d[0] < -0.01
		on[3] = d[0] > 0.01
	} else {
		on[0] = ebiten.IsKeyPressed(ebiten.KeyArrowUp) || ebiten.IsKeyPressed(ebiten.KeyW)
		on[1] = ebiten.IsKeyPressed(ebiten.KeyArrowDown) || ebiten.IsKeyPressed(ebiten.KeyS)
		on[2] = ebiten.IsKeyPressed(ebiten.KeyArrowLeft) || ebiten.IsKeyPressed(ebiten.KeyA)
		on[3] = ebiten.IsKeyPressed(ebiten.KeyArrowRight) || ebiten.IsKeyPressed(ebiten.KeyD)
	}

	// Порядок збігається з on[]: up, down, left, right.
	dirs := [4][2]float32{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}

	active := color.RGBA{240, 200, 90, 255} // бурштиновий — як рамка рівня
	idle := color.RGBA{62, 92, 110, 255}    // тьмяний, у тон води (без альфи: RGBA тут
	//                                          премножена, і напівпрозорість давала б
	//                                          перепалений відтінок — див. drawFur)

	cx, cy := float32(inputWidgetX), float32(inputWidgetY)
	for i, d := range dirs {
		col := idle
		if on[i] {
			col = active
		}
		x0 := cx + d[0]*inputStickGap
		y0 := cy + d[1]*inputStickGap
		x1 := cx + d[0]*(inputStickGap+inputStickLen)
		y1 := cy + d[1]*(inputStickGap+inputStickLen)
		vector.StrokeLine(screen, x0, y0, x1, y1, inputStickWidth, col, true)
	}
}

// [ПАЛІТРА] Кольори стін — в одному місці, а не магічними числами в циклі
// малювання. Ними ж фарбується межа рівня: вона теж стіна.
var (
	wallFill = color.RGBA{55, 55, 75, 255}  // тіло тайла
	wallEdge = color.RGBA{80, 80, 110, 255} // світліший контур — дає обʼєм
)

// drawSea малює фон-море: вертикальний градієнт від світлішого верху до темнішої
// глибини. Смугами, а не пікселями — seaBands штук FillRect на кадр, тобто дешевше
// за один намальований юніт.
//
// Чому градієнт, а не screen.Fill одним кольором: плоска заливка читається як
// «пофарбоване тло», а вертикальний перехід — як ГЛИБИНА, і цього досить, щоб поле
// перестало бути абстрактним аркушем. Текстуру можна буде покласти згори пізніше,
// нічого тут не переписуючи.
func drawSea(screen *ebiten.Image) {
	h := float32(screenHeight) / seaBands
	for i := 0; i < seaBands; i++ {
		t := float32(i) / (seaBands - 1) // 0 = верх, 1 = глибина
		col := color.RGBA{
			R: uint8(float32(seaTopR) + (seaBotR-seaTopR)*t),
			G: uint8(float32(seaTopG) + (seaBotG-seaTopG)*t),
			B: uint8(float32(seaTopB) + (seaBotB-seaTopB)*t),
			A: 255,
		}
		// +1 до висоти: щоб між смугами не лишалось волосяних щілин через округлення.
		vector.FillRect(screen, 0, float32(i)*h, screenWidth, h+1, col, false)
	}
}

// drawText малює текст відцентровано відносно точки (cx, cy).
// [GO: ColorScale] — гліфи шрифту білі за замовчуванням.
// Scale(r,g,b,a) множить кольори: Scale(0,0,0,1) → чорний текст.
func drawText(screen *ebiten.Image, str string, size, cx, cy float64, clr color.RGBA) {
	face := &etext.GoTextFace{Source: fontFaceSource, Size: size}
	w, h := etext.Measure(str, face, 0)
	op := &etext.DrawOptions{}
	op.GeoM.Translate(cx-w/2, cy-h/2)
	op.ColorScale.Scale(
		float32(clr.R)/255,
		float32(clr.G)/255,
		float32(clr.B)/255,
		float32(clr.A)/255,
	)
	etext.Draw(screen, str, face, op)
}

// drawTextL — як drawText, але x = ЛІВИЙ край тексту (не центр). Для панелей/
// таблиць із рядками різної довжини: текст не «розповзається» за межі й не
// вилазить за край екрана незалежно від довжини рядка.
func drawTextL(screen *ebiten.Image, str string, size, x, y float64, clr color.RGBA) {
	face := &etext.GoTextFace{Source: fontFaceSource, Size: size}
	_, h := etext.Measure(str, face, 0)
	op := &etext.DrawOptions{}
	op.GeoM.Translate(x, y-h/2)
	op.ColorScale.Scale(
		float32(clr.R)/255,
		float32(clr.G)/255,
		float32(clr.B)/255,
		float32(clr.A)/255,
	)
	etext.Draw(screen, str, face, op)
}

// drawFur — [ВОРС] хутро юніта: 2 сегменти на ворсинку (корінь→середина→кінчик).
// Малюємо ПЕРЕД тілом, щоб корені ховались під квадратом.
//
// [GO/EBITEN: ПРЕМНОЖЕНА АЛЬФА] color.RGBA тут трактується як premultiplied:
// щоб отримати напівпрозорий колір, RGB треба помножити на частку альфи, а не
// просто зменшити A — інакше вийде перепалений відтінок.
func drawFur(screen *ebiten.Image, p *Pixel) {
	const a = 165
	col := color.RGBA{
		R: uint8(int(p.Color.R) * a / 255),
		G: uint8(int(p.Color.G) * a / 255),
		B: uint8(int(p.Color.B) * a / 255),
		A: a,
	}
	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2
	for i := 0; i < furStrands; i++ {
		mx, my := p.Fur[i][0][0], p.Fur[i][0][1]
		tx, ty := p.Fur[i][1][0], p.Fur[i][1][1]
		vector.StrokeLine(screen, cx, cy, mx, my, 2, col, false)
		vector.StrokeLine(screen, mx, my, tx, ty, 1, col, false)
	}
}

// bodyRestRadius — радіус «спокійного» контуру вздовж напрямку dirs8[i].
//
// Відстань від центра до контуру КВАДРАТА з півстороною R уздовж одиничного
// напрямку (dx,dy) — це R / max(|dx|,|dy|): для осей виходить R, для діагоналей
// R√2. Тобто вісім таких радіусів відтворюють наш квадрат ТОЧНО, до пікселя, — і
// перехід на восьмикутник у спокої не змінює картинку взагалі. Змінюється лише те,
// що тепер контур є що деформувати.
func bodyRestRadius(i int) float32 {
	dx, dy := dirs8[i][0], dirs8[i][1]
	m := dx
	if m < 0 {
		m = -m
	}
	ady := dy
	if ady < 0 {
		ady = -ady
	}
	if ady > m {
		m = ady
	}
	return (pixelSize / 2) / m
}

// drawBody малює тіло як восьмикутник, деформований НАМІРОМ мережі.
//
// Вісім Q-значень лягають на вісім радіусів: тіло тягнеться туди, куди агент хоче,
// і підбирається з протилежного боку. tanh, а не лінійна нормалізація — навмисно:
// нормалізація на розмах зробила б навіть мікроскопічну різницю в Q максимальною
// деформацією і ЗАХОВАЛА б головний діагностичний випадок. З tanh пласка Q дає нуль
// відхилення, тобто рівний квадрат: «мережа не розрізняє дій» видно оком.
// Великий розмах Q (у вбивці нагороди більші) плавно насичується замість вибуху.
//
// Масштаб від пружини (updateBody) — скаляр, він не конфліктує з напрямком форми:
// розмір говорить про рух, форма — про намір.
func drawBody(screen *ebiten.Image, p Pixel, col color.RGBA) {
	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2

	scale := p.BodyScale
	if scale <= 0 {
		scale = 1 // юніт, створений в обхід resetFur (тести)
	}

	// Середнє Q — точка відліку: цікавить ПЕРЕВАГА дії над іншими, а не абсолют.
	var mean float32
	hasQ := p.Brain != nil
	if hasQ {
		for i := 0; i < brainActions; i++ {
			mean += p.Brain.lastQ[i]
		}
		mean /= brainActions
	}

	// Вісім вершин контуру.
	var vx, vy [brainActions]float32
	for i := 0; i < brainActions; i++ {
		r := bodyRestRadius(i) * scale
		if hasQ {
			r *= 1 + bodyQStretch*tanh((p.Brain.lastQ[i]-mean)/bodyQScale)
		}
		vx[i] = cx + dirs8[i][0]*r
		vy[i] = cy + dirs8[i][1]*r
	}

	// [ОРГАНІКА] Зʼєднуємо вершини НЕ прямими, а квадратичними кривими: сама
	// вершина стає контрольною точкою, а крива проходить через СЕРЕДИНИ ребер.
	// Пряме зʼєднання давало гранчастий контур, і коли одна Q переважала сусідні,
	// вилазив гострий шпиль — тіло читалось як «квадрат, від якого відкусили».
	// Тепер сплеск Q дає плавну випуклість, а не колючку.
	//
	// Побічний ефект, який тут доречний: крива зрізає кути, тож у спокої контур
	// стає не строгим квадратом, а квадратом зі скругленими кутами — саме те, що
	// треба для істоти, а не для тайла.
	mid := func(i, j int) (float32, float32) {
		return (vx[i] + vx[j]) / 2, (vy[i] + vy[j]) / 2
	}
	path := &vector.Path{}
	sx, sy := mid(brainActions-1, 0)
	path.MoveTo(sx, sy)
	for i := 0; i < brainActions; i++ {
		nx, ny := mid(i, (i+1)%brainActions)
		path.QuadTo(vx[i], vy[i], nx, ny) // вершина = контрольна точка
	}
	path.Close()

	var op vector.DrawPathOptions
	op.AntiAlias = true // контур органічний, без згладжування виглядав би рваним
	op.ColorScale.ScaleWithColor(col)
	vector.FillPath(screen, path, nil, &op)
}

// drawPixel малює тіло з flash-ефектом, HP bar і міткою.
func drawPixel(screen *ebiten.Image, p Pixel) {
	// Flash: поки HitTimer > 0 — малюємо білим
	col := p.Color
	if p.HitTimer > 0 {
		col = color.RGBA{255, 255, 255, 255}
	}
	drawBody(screen, p, col)

	// HP bar — тільки для ворогів (MaxHP > 0)
	if p.MaxHP > 0 {
		const barH = 3
		// Ширина смуги йде за пружиною тіла: інакше при стисканні вона стирчала б
		// з боків ширшою за самого юніта.
		bw := float32(pixelSize)
		if p.BodyScale > 0 {
			bw *= p.BodyScale
		}
		barX := p.X + (pixelSize-bw)/2
		barY := p.Y - barH - 1
		vector.FillRect(screen, barX, barY, bw, barH, color.RGBA{80, 0, 0, 200}, false)
		hpRatio := float32(p.HP) / float32(p.MaxHP)
		filled := bw * hpRatio
		barColor := color.RGBA{
			R: uint8(255 * (1 - hpRatio)),
			G: uint8(220 * hpRatio),
			B: 0, A: 255,
		}
		vector.FillRect(screen, barX, barY, filled, barH, barColor, false)
	}

	if p.Label != "" {
		cx := float64(p.X) + pixelSize/2
		cy := float64(p.Y) + pixelSize/2
		drawText(screen, p.Label, labelFontSize, cx, cy, color.RGBA{0, 0, 0, 255})
	}
}

// drawBrainSensors візуалізує «під капотом» Q-learner-а:
//   - 8 whiskers: лінія в кожен напрямок, довжина = вільний простір до стіни,
//     колір від зеленого (чисто) до червоного (стіна близько);
//   - жовта стрілка — напрямок дії, яку мережа щойно обрала.
func drawBrainSensors(screen *ebiten.Image, e Pixel, cx, cy float32) {
	for i := 0; i < brainWhiskers; i++ {
		w := e.Brain.lastWhiskers[i]
		length := float32(whiskerRange)
		if w > 0 {
			length = whiskerRange * (1 - w) // відстань до стіни
		}
		ex := cx + dirs8[i][0]*length
		ey := cy + dirs8[i][1]*length
		col := color.RGBA{R: uint8(60 + 195*w), G: uint8(200 * (1 - w)), B: 60, A: 150}
		vector.StrokeLine(screen, cx, cy, ex, ey, 1, col, false)
	}
	// Обрана дія — яскрава жовта стрілка.
	a := e.Brain.lastAction
	vector.StrokeLine(screen, cx, cy, cx+dirs8[a][0]*45, cy+dirs8[a][1]*45, 2,
		color.RGBA{255, 255, 0, 255}, false)
}

// Draw малює поточний стан на екрані.
func (g *Game) Draw(screen *ebiten.Image) {
	if g.firstPerson {
		// [RAYCASTER] Вид від першої особи замість топ-дауну (клавіша F).
		g.drawFirstPerson(screen)
	} else {
		drawSea(screen) // [ФОН] море: вертикальний градієнт замість плоскої заливки

		// Малюємо тайли рівня.
		// Стіни — темно-сірі, підлога — не малюється (фон і є підлогою).
		for row := 0; row < boidMapH; row++ {
			for col := 0; col < boidMapW; col++ {
				if tileMap[row][col] {
					x := float32(col * pixelSize)
					y := float32(row * pixelSize)
					vector.FillRect(screen, x, y, pixelSize, pixelSize, wallFill, false)
					// Тонкий контур стіни для об'єму
					vector.StrokeRect(screen, x, y, pixelSize, pixelSize, 1, wallEdge, false)
				}
			}
		}

		// [МЕЖА РІВНЯ] Рамка кольором СТІН — бо межа і є стіна: юніти від неї
		// відбиваються, гравець упирається.
		//
		// Раніше вона була бурштиновою й позначала wrap-перехід («гравець протікає на
		// інший бік»). Wrap-around прибрано давно, а колір лишився й читався як border
		// з верстки — позначка механіки, якої вже немає.
		vector.StrokeRect(screen, 1, 1, screenWidth-2, screenHeight-2, 2, wallEdge, false)

		// [СТИГМЕРГІЯ] Теплова карта феромонів фрустрації (під ворогами).
		// Чим яскравіше-червоніше — тим сильніший слід «тут застрягали».
		if showFrustration && pheromonesEnabled {
			for row := 0; row < boidMapH; row++ {
				for col := 0; col < boidMapW; col++ {
					f := g.frustration[row][col]
					if f <= 0.05 {
						continue
					}
					a := f * 12
					if a > 140 {
						a = 140
					}
					x := float32(col * pixelSize)
					y := float32(row * pixelSize)
					vector.FillRect(screen, x, y, pixelSize, pixelSize, color.RGBA{255, 80, 0, uint8(a)}, false)
				}
			}
		}

		// [FLOW-FIELD] Поле напрямків до гравця (клавіша V) — під юнітами,
		// щоб стрілки не перекривали ворогів і гравця.
		if showFlowField != 0 {
			g.drawFlowField(screen)
		}

		for i, e := range g.units {
			// Радіус огляду — дуже прозоре кільце навколо ворога
			cx := e.X + pixelSize/2
			cy := e.Y + pixelSize/2
			if showDetectionCircle {
				vector.StrokeCircle(screen, cx, cy, e.Cfg.DetectionRange, 1, color.RGBA{255, 255, 255, 5}, false)
			}
			if showWhiskers && e.Brain != nil {
				drawBrainSensors(screen, e, cx, cy)
			}
			drawFur(screen, &g.units[i])
			drawPixel(screen, e)
		}
		drawFur(screen, &g.player)
		drawPixel(screen, g.player)

		// Кільце атаки — StrokeCircle (контур) замість DrawFilledCircle (заповнене).
		// Виглядає чистіше як індикатор зони удару і не засліплює.
		// alpha = 180..0 — плавно зникає разом з attackTimer.
		if g.attackTimer > 0 {
			cx := g.player.X + pixelSize/2
			cy := g.player.Y + pixelSize/2
			alpha := uint8(180 * g.attackTimer / attackDuration)
			vector.StrokeCircle(screen, cx, cy, attackRadius, 2, color.RGBA{255, 255, 80, alpha}, true)
		}
	} // кінець топ-даун-гілки

	// HUD
	level := g.tick/levelUpEvery + 1
	playerSpeed := math.Sqrt(float64(g.player.VelX*g.player.VelX + g.player.VelY*g.player.VelY))
	white := color.RGBA{255, 255, 255, 255}
	cyan := color.RGBA{0, 220, 180, 255}
	fps := ebiten.ActualFPS()
	drawText(screen, fmt.Sprintf("LVL %d", level), 10, 30, 10, white)

	// [БІЙ] HP гравця — червоніє, коли мало. Видно і у виді від 1-ї особи.
	hpCol := color.RGBA{90, 230, 120, 255}
	if g.player.MaxHP > 0 && g.player.HP*3 <= g.player.MaxHP {
		hpCol = color.RGBA{240, 80, 60, 255}
	}
	drawText(screen, fmt.Sprintf("HP %d/%d", g.player.HP, g.player.MaxHP), 10, 110, 10, hpCol)
	drawText(screen, fmt.Sprintf("SPD %.1f", playerSpeed), 10, screenWidth-32, 10, cyan)
	drawText(screen, fmt.Sprintf("DIF %.1f", g.difficulty), 10, screenWidth-32, 25, color.RGBA{255, 140, 50, 255})
	drawText(screen, fmt.Sprintf("FPS %.0f", fps), 10, screenWidth/2, 10, color.RGBA{150, 150, 150, 255})

	// [ЗАМІРИ] TPS окремо від FPS — це РІЗНІ речі, і плутанина між ними вже раз
	// зіпсувала висновок. FPS = частота МАЛЮВАННЯ (Draw, темп монітора), TPS =
	// частота ЛОГІКИ (Update, наші тіки). Коли важчає обчислення (наприклад BPTT
	// у GRU), просідає TPS, а FPS лишається 120 — і на скріні все «нормально».
	// Червоний = логіка не встигає за цільовим темпом: годинниковий час на скріні
	// більше не перекладається в тіки один в один.
	tps := ebiten.ActualTPS()
	tpsCol := color.RGBA{150, 150, 150, 255}
	if tps < float64(ebiten.TPS())*0.9 {
		tpsCol = color.RGBA{240, 120, 60, 255}
	}
	drawText(screen, fmt.Sprintf("TPS %.0f", tps), 10, screenWidth/2, 25, tpsCol)

	if showInput {
		g.drawInputDirs(screen)
	}

	// [ПАУЗА] Обовʼязково видимо: без підпису застиглий кадр не відрізнити від
	// зависання гри.
	if g.paused {
		drawText(screen, "PAUSED  (P)", gameOverFontSize*0.8,
			float64(screenWidth)/2, float64(screenHeight)/2,
			color.RGBA{255, 220, 50, 255})
	}

	if g.gameOver {
		cx := float64(screenWidth) / 2
		cy := float64(screenHeight) / 2
		yellow := color.RGBA{255, 220, 50, 255}
		gray := color.RGBA{180, 180, 180, 255}

		level := g.tick/levelUpEvery + 1
		points := g.tick * 1000 / 60

		drawText(screen, "GAME OVER", gameOverFontSize, cx, cy-36, white)
		drawText(screen, fmt.Sprintf("Level: %d   Points: %d", level, points), gameOverFontSize*0.7, cx, cy-12, yellow)
		drawText(screen, "Use arrows / WASD to escape!", gameOverFontSize*0.6, cx, cy+10, gray)
		drawText(screen, "R - restart    ESC - exit", gameOverFontSize*0.6, cx, cy+28, white)
	}

	// [МЕТРИКИ] Панель кривої навчання (клавіша G) — поверх усього.
	if showMetrics {
		g.metrics.draw(screen)
	}
}
