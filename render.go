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
		// [УХИЛЕННЯ] У девʼятої дії напрямку немає — хрест лишається порожнім.
		if a := g.player.Brain.lastAction; a < brainWhiskers {
			d := dirs8[a]
			on[0] = d[1] < -0.01
			on[1] = d[1] > 0.01
			on[2] = d[0] < -0.01
			on[3] = d[0] > 0.01
		}
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
		vector.StrokeLine(screen, x0, y0, x1, y1, inputStickWidth, col, antiAlias)
	}
}

// [ПАЛІТРА] Кольори стін — в одному місці, а не магічними числами в циклі
// малювання. Ними ж фарбується межа рівня: вона теж стіна.
var (
	wallFill = color.RGBA{55, 55, 75, 255}  // тіло тайла
	wallEdge = color.RGBA{80, 80, 110, 255} // світліший контур — дає обʼєм
)

// ══════════════════════════════════════════════════════════════════════════
// [ПАКЕТУВАННЯ] ДВА ШЛЯХИ ДО GPU, І ВОНИ НЕ ВЗАЄМОЗАМІННІ
//
// У ebiten 2.9 малювання вектора йде двома різними механізмами, і сплутати їх коштує
// вдвічі більше кадру. Записую обидва, бо перевіряли на собі.
//
//	FillPath / StrokePath  — НЕ малюють одразу. Накопичують шляхи в спільному стані
//	                         (fillPathsState) і віддають одним DrawTriangles, доки не
//	                         зміняться antialias, blend або fillRule. Растеризація
//	                         йде через АТЛАС — дорого за шлях, дешево за виклик.
//	FillRect / FillCircle  — без згладжування йдуть повз цей стан, прямо в
//	StrokeLine / …            DrawTriangles32: два-три десятки трикутників, і все.
//	                         Дешево за фігуру, але СКИДАЄ накопичений батч шляхів.
//
// ЗВІДСИ ПРАВИЛО, і воно НЕ «все через шлях»:
//
//	складна геометрія (ворс, кінцівки, відросток, тіло, стіни) → шлях, і бажано
//	   один шлях на групу: там виграш саме у кількості викликів;
//	проста фігура (коло, прямокутник) → лишається негайною: як шлях вона стає
//	   растеризацією в атлас замість двох трикутників.
//
// Ми це перевірили навпаки й програли. Переведення всіх кульок, долонь, смуг HP і
// смуг моря на FillPath дало 30 FPS замість 63 — рівно вдвічі гірше. CPU при цьому не
// змінився взагалі (0.441 мс проти 0.453 мс на 500 фігур), тобто ціна цілком на боці
// GPU: 500 растеризацій шляху замість 500 пар трикутників.
//
// ЩО ЛИШАЄТЬСЯ ПРАВДОЮ: змішувати їх упереміш — найгірше з двох. Кожна проста фігура
// між шляхами рве батч. Правильна відповідь тут не «переписати фігури», а не
// перемішувати порядок малювання; це ще не зроблено й лишається наступним кроком.
//
// ТОЙ САМИЙ МЕХАНІЗМ ПОЯСНЮЄ ЦІНУ ЗГЛАДЖУВАННЯ. Коли AA стояв лише на тілі, кожен
// юніт перемикав antialias з false на true й назад — тобто скидав батч двічі, ще й
// заводив окремий офскрін-буфер подвійного розміру (див. antiAlias у main.go). Тому
// прапорець тепер один на весь рендер: або згладжуємо все, або нічого.
// ══════════════════════════════════════════════════════════════════════════

// pathOpts — спільні опції малювання шляху. Єдине місце, де читається antiAlias:
// якби кожна функція вирішувала сама, вони б розʼїхались і батч рвався б знову.
func pathOpts(col color.RGBA) vector.DrawPathOptions {
	var op vector.DrawPathOptions
	op.AntiAlias = antiAlias
	op.ColorScale.ScaleWithColor(col)
	return op
}

// wallPath — шлях стін, що ПЕРЕВИКОРИСТОВУЄТЬСЯ між кадрами.
//
// Пакетна змінна, а не локальна: інакше кожен кадр алокував би шлях на кілька сотень
// тайлів і віддавав його збирачу сміття. Reset() лишає ємність.
var wallPath vector.Path

// buildWallPath складає всі ВИДИМІ тайли стін в один шлях і повертає їхню кількість.
//
// Окремою функцією з тієї самої причини, що bodyVertices: малювання йде прямо в
// ebiten і назовні не віддає нічого, тож перевірити відсікання тестом можна лише так.
// Повернена кількість — не для малювання, вона існує рівно заради тесту.
func buildWallPath(p *vector.Path) int {
	n := 0
	for row := 0; row < boidMapH; row++ {
		for col := 0; col < boidMapW; col++ {
			if !tileMap[row][col] {
				continue
			}
			x := float32(col * pixelSize)
			y := float32(row * pixelSize)
			// [ЗУМ] За кадром — не платимо за те, чого не видно. Предикат ТОЙ САМИЙ, що
			// для юнітів, хоча тайлу його запас завеликий (він розрахований на ворс).
			// Це свідомо: одне правило відсікання на весь рендер, а зайвий запас тепер
			// коштує лише вершин, а не викликів, — саме заради цього й пакетуємо.
			if !cam.visible(x, y) {
				continue
			}
			sx, sy, s := cam.px(x), cam.py(y), cam.s(pixelSize)
			p.MoveTo(sx, sy)
			p.LineTo(sx+s, sy)
			p.LineTo(sx+s, sy+s)
			p.LineTo(sx, sy+s)
			p.Close() // замкнений підшлях = той самий контур, що давав StrokeRect
			n++
		}
	}
	return n
}

// drawSea малює фон-море: вертикальний градієнт від світлішого верху до темнішої
// глибини. Смугами, а не пікселями — seaBands штук FillRect на кадр, тобто дешевше
// за один намальований юніт.
//
// Чому градієнт, а не screen.Fill одним кольором: плоска заливка читається як
// «пофарбоване тло», а вертикальний перехід — як ГЛИБИНА, і цього досить, щоб поле
// перестало бути абстрактним аркушем. Текстуру можна буде покласти згори пізніше,
// нічого тут не переписуючи.
//
// [КАМЕРА] Смуги прив'язані до СВІТОВОГО Y, а не до екранного. Поки світ дорівнював
// вікну, різниці не було. Коли світ став удвічі вищим за вікно, екранна прив'язка
// перетворила глибину на скайбокс: та сама світова клітинка світлішала й темнішала
// залежно від того, де стоїть камера, тобто градієнт перестав щось означати.
//
// Зі світовою прив'язкою верх карти читається як мілина, низ — як глибина, і колір
// клітинки не залежить від камери. Наслідок, який видно оком: рухаючись вертикально,
// ти справді «занурюєшся». Ціна — за один кадр видно лише частину діапазону (при
// зумі 1.0 половину, при 4× — восьму), тож повний розкид кольору мусив вирости
// вдвічі, а seaBands — теж удвічі, щоб смуги лишились такими ж дрібними на екрані
// (див. sea* у tuning_visual.go).
func drawSea(screen *ebiten.Image) {
	bandH := float32(worldHeight) / seaBands // висота смуги у СВІТОВИХ пікселях
	for i := 0; i < seaBands; i++ {
		// +1 екранний піксель: щоб між смугами не лишалось волосяних щілин
		// через округлення. Саме екранний, а не світовий — щілина народжується
		// при растеризації, тож і запас потрібен у пікселях екрана.
		sy := cam.py(float32(i) * bandH)
		h := cam.s(bandH) + 1
		if sy+h < 0 || sy > screenHeight {
			continue // [КАМЕРА] смуга за кадром — не платимо за те, чого не видно
		}
		t := float32(i) / (seaBands - 1) // 0 = поверхня, 1 = глибина
		col := color.RGBA{
			R: uint8(float32(seaTopR) + (seaBotR-seaTopR)*t),
			G: uint8(float32(seaTopG) + (seaBotG-seaTopG)*t),
			B: uint8(float32(seaTopB) + (seaBotB-seaTopB)*t),
			A: 255,
		}
		vector.FillRect(screen, 0, sy, screenWidth, h, col, false)
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
	fx, fy := furRoot(cx, cy, bodyScaleOf(p))

	// [ПАКЕТУВАННЯ] Замість StrokeLine на КОЖЕН сегмент (8×furJoints викликів) збираємо
	// сегменти в спільні шляхи за товщиною й малюємо один StrokePath на групу.
	//
	// Товщина в StrokeOptions одна на шлях, тому групи й потрібні: неперервне звуження
	// довелось би малювати посегментно. Три сходинки на волосині 2→1px оком не
	// відрізнити від плавного переходу, а викликів стає в furJoints/furWidthGroups разів
	// менше — при 14 колінах це 37×.
	var paths [furWidthGroups]vector.Path
	for i := 0; i < furStrands; i++ {
		px, py := fx, fy
		for j := 0; j < furJoints; j++ {
			g := j * furWidthGroups / furJoints
			jx, jy := p.Fur[i][j][0], p.Fur[i][j][1]
			paths[g].MoveTo(cam.px(px), cam.py(py))
			paths[g].LineTo(cam.px(jx), cam.py(jy))
			px, py = jx, jy
		}
	}
	op := pathOpts(col)
	for g := 0; g < furWidthGroups; g++ {
		t := float32(g) / float32(furWidthGroups-1)
		vector.StrokePath(screen, &paths[g],
			&vector.StrokeOptions{Width: cam.s(furWidthRoot + (furWidthTip-furWidthRoot)*t)}, &op)
	}
}

// drawBalls — [КУЛЬКИ] дві кульки, що бовтаються під тілом.
//
// Малюємо ПІСЛЯ тіла, а не до: вони мають лишатись видимими цілком, коли підтягуються
// під корпус. Ворс навпаки йде до тіла, щоб корені ховались — різні цілі, різний порядок.
//
// Колір — тіла, без прозорості: кульки читаються як частина істоти, а не як ефект.
func drawBalls(screen *ebiten.Image, p *Pixel) {
	for i := 0; i < ballCount; i++ {
		vector.FillCircle(screen, cam.px(p.Balls[i][0]), cam.py(p.Balls[i][1]), cam.s(ballRadius), p.Color, antiAlias)
	}
}

// drawLimbs — [КІНЦІВКИ] чотири короткі лінії від кутів корпуса з кулькою-долонею.
//
// Малюємо ПЕРЕД тілом: кріплення мусить ховатись під корпусом, інакше видно, що лінія
// починається в порожнечі. Ворс іде так само й з тієї ж причини; кульки й щупальце
// навпаки — після, бо вони мають лишатись видимими цілком.
func drawLimbStrokes(screen *ebiten.Image, p *Pixel) {
	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2
	// Усі чотири кінцівки — один шлях: товщина в них однакова, тож ділити нема на що.
	var path vector.Path
	for i := 0; i < limbCount; i++ {
		rx, ry := limbRoot(i, cx, cy, bodyScaleOf(p))
		path.MoveTo(cam.px(rx), cam.py(ry))
		path.LineTo(cam.px(p.Limbs[i][0]), cam.py(p.Limbs[i][1]))
	}
	op := pathOpts(p.Color)
	vector.StrokePath(screen, &path, &vector.StrokeOptions{Width: cam.s(limbWidth)}, &op)
}

// drawLimbTips — кульки-долоні. Окремо від штрихів: вони проста фігура, а не шлях,
// і йдуть у другому проході (див. [ДВА ПРОХОДИ] у Draw).
func drawLimbTips(screen *ebiten.Image, p *Pixel) {
	for i := 0; i < limbCount; i++ {
		vector.FillCircle(screen, cam.px(p.Limbs[i][0]), cam.py(p.Limbs[i][1]), cam.s(limbTipDot), p.Color, false)
	}
}

// drawTentacle — [ВІДРОСТОК] ламана від низу тіла крізь суглоби, з кулькою на кінці.
//
// Малюємо ПЕРЕД кульками, але ПІСЛЯ тіла: відросток має виходити з-під корпуса, а
// кульки лишатись поверх усього.
func drawTentacleStroke(screen *ebiten.Image, p *Pixel) {
	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2
	px, py := tentRoot(cx, cy, bodyScaleOf(p))
	var path vector.Path
	path.MoveTo(cam.px(px), cam.py(py))
	// Та сама межа, що у фізиці (updateTentacle) — через tentJointsOf. Розʼїзд тут
	// означав би або намальований суглоб, який ніхто не рухає, або живий, якого не видно.
	for i, n := 0, tentJointsOf(p); i < n; i++ {
		path.LineTo(cam.px(p.Tent[i][0]), cam.py(p.Tent[i][1]))
		px, py = p.Tent[i][0], p.Tent[i][1]
	}
	op := pathOpts(p.Color)
	vector.StrokePath(screen, &path, &vector.StrokeOptions{Width: cam.s(tentWidth)}, &op)
}

// drawTentacleTip — кулька на кінці відростка. Кінець ланцюжка перераховуємо тут
// заново, а не тягнемо з drawTentacleStroke: інакше довелось би вертати координату
// через параметр і два проходи стали б звʼязаними.
func drawTentacleTip(screen *ebiten.Image, p *Pixel) {
	n := tentJointsOf(p)
	px, py := tentRoot(p.X+pixelSize/2, p.Y+pixelSize/2, bodyScaleOf(p))
	if n > 0 {
		px, py = p.Tent[n-1][0], p.Tent[n-1][1]
	}
	vector.FillCircle(screen, cam.px(px), cam.py(py), cam.s(tentTipBall), p.Color, false)
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
// bodyVertices — вісім вершин контуру тіла. Винесено з drawBody НЕ заради краси:
// саме тут жила вада, коли масив мав розмір brainActions (9), а заповнювався на 8 —
// девʼята вершина лишалась у початку координат, і тіло тягнулось у кут екрана.
//
// Окремою функцією, бо це єдиний спосіб перевірити геометрію тестом: малювання
// відразу йде в ebiten і назовні нічого не віддає.
//
// Розмір масиву — brainWhiskers (кількість НАПРЯМКІВ), а не brainActions (кількість
// ДІЙ). З появою ухилення це різні числа, і плутати їх не можна ніде.
func bodyVertices(p Pixel, cx, cy, scale float32) (vx, vy [brainWhiskers]float32) {
	var mean float32
	hasQ := p.Brain != nil
	if hasQ {
		for i := 0; i < brainWhiskers; i++ {
			mean += p.Brain.lastQ[i]
		}
		mean /= brainWhiskers
	}
	for i := 0; i < brainWhiskers; i++ {
		r := bodyRestRadius(i) * scale
		if hasQ {
			r *= 1 + bodyQStretch*tanh((p.Brain.lastQ[i]-mean)/bodyQScale)
		}
		vx[i] = cx + dirs8[i][0]*r
		vy[i] = cy + dirs8[i][1]*r
	}
	return vx, vy
}

func drawBody(screen *ebiten.Image, p Pixel, col color.RGBA) {
	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2

	scale := p.BodyScale
	if scale <= 0 {
		scale = 1 // юніт, створений в обхід resetFur (тести)
	}

	vx, vy := bodyVertices(p, cx, cy, scale)

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
	sx, sy := mid(brainWhiskers-1, 0)
	path.MoveTo(cam.px(sx), cam.py(sy))
	for i := 0; i < brainWhiskers; i++ {
		nx, ny := mid(i, (i+1)%brainWhiskers)
		path.QuadTo(cam.px(vx[i]), cam.py(vy[i]), cam.px(nx), cam.py(ny)) // вершина = контрольна
	}
	path.Close()

	// Через pathOpts, а не вручну: колір і згладжування мусять братись з одного місця.
	// Тут уже був цей баг — я прибрав рядок ColorScale, вважаючи, що op приходить із
	// pathOpts, а він лишався нульовим. Нульовий ColorScale не «без кольору», а
	// БІЛИЙ: він множить, тож одиничний масштаб дає білі гліфи тіла при живому ворсі.
	op := pathOpts(col)
	vector.FillPath(screen, path, nil, &op)
}

// drawPixel малює тіло з flash-ефектом, HP bar і міткою.
// drawDash — [РИВОК] телеграф атаки. Це не оздоблення: уся механіка тримається на
// тому, що ЛІНІЮ УДАРУ видно заздалегідь. Не намалювати її означало б зробити атаку
// невідворотною знову, тобто повернути ульту під іншим імʼям.
//
//	замах  — лінія в замкненому напрямку, наливається кольором і ДОВЖИНОЮ: видно не
//	         лише «зараз ударю», а й куди дістане (dashActive × швидкість ривка);
//	ривок  — яскравий слід уздовж пройденого відрізка;
//	відхід — кільце, що гасне: видно, скільки ще бути безпорадним.
//
// Юніт через свою ознаку бачить лише замах (dashWindupProgress) — людині корисніше
// бачити всі три фази, бо вона ще й цілиться.
func drawDash(screen *ebiten.Image, p *Pixel) {
	if p.DashPhase == dashIdle {
		return
	}
	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2
	reach := float32(dashActive) * playerBaseSpeed * dashSpeedMulti

	switch p.DashPhase {
	case dashPhaseWindup:
		f := dashWindupProgress(p)
		ln := reach * f
		vector.StrokeLine(screen, cam.px(cx), cam.py(cy), cam.px(cx+p.DashDirX*ln), cam.py(cy+p.DashDirY*ln),
			cam.s(1+2*f), color.RGBA{255, uint8(220 - 140*f), 60, uint8(90 + 150*f)}, antiAlias)
	case dashPhaseActive:
		vector.StrokeLine(screen, cam.px(cx-p.DashDirX*reach), cam.py(cy-p.DashDirY*reach), cam.px(cx), cam.py(cy),
			cam.s(4), color.RGBA{255, 255, 200, 210}, antiAlias)
	default:
		a := uint8(120 * p.DashTimer / dashRecovery)
		vector.StrokeCircle(screen, cam.px(cx), cam.py(cy), cam.s(pixelSize*0.9), cam.s(1.5), color.RGBA{120, 170, 255, a}, antiAlias)
	}
}

func drawPixelBody(screen *ebiten.Image, p Pixel) {
	// Flash: поки HitTimer > 0 — малюємо білим
	col := p.Color
	if p.HitTimer > 0 {
		col = color.RGBA{255, 255, 255, 255}
	}
	drawBody(screen, p, col)
}

// drawPixelOverlay — смуга HP і мітка: прості фігури й текст, тобто другий прохід.
//
// Побічний виграш, який видно оком: смуга тепер завжди ЗВЕРХУ. Раніше сусідній юніт,
// намальований пізніше, міг її перекрити — і здоровʼя ставало нечитним саме в купі,
// тобто рівно тоді, коли воно потрібне.
func drawPixelOverlay(screen *ebiten.Image, p Pixel) {
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
		vector.FillRect(screen, cam.px(barX), cam.py(barY), cam.s(bw), cam.s(barH), color.RGBA{80, 0, 0, 200}, false)
		hpRatio := float32(p.HP) / float32(p.MaxHP)
		filled := bw * hpRatio
		barColor := color.RGBA{
			R: uint8(255 * (1 - hpRatio)),
			G: uint8(220 * hpRatio),
			B: 0, A: 255,
		}
		vector.FillRect(screen, cam.px(barX), cam.py(barY), cam.s(filled), cam.s(barH), barColor, false)
	}

	if p.Label != "" {
		cx := float64(cam.px(p.X + pixelSize/2))
		cy := float64(cam.py(p.Y + pixelSize/2))
		drawText(screen, p.Label, labelFontSize*float64(cam.zoom), cx, cy, color.RGBA{0, 0, 0, 255})
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
		vector.StrokeLine(screen, cam.px(cx), cam.py(cy), cam.px(ex), cam.py(ey), cam.s(1), col, false)
	}
	// Обрана дія — яскрава жовта стрілка.
	a := e.Brain.lastAction
	vector.StrokeLine(screen, cam.px(cx), cam.py(cy), cam.px(cx+dirs8[a][0]*45), cam.py(cy+dirs8[a][1]*45),
		cam.s(2), color.RGBA{255, 255, 0, 255}, false)
}

// Draw малює поточний стан на екрані.
func (g *Game) Draw(screen *ebiten.Image) {
	if g.firstPerson {
		// [RAYCASTER] Вид від першої особи замість топ-дауну (клавіша F).
		g.drawFirstPerson(screen)
	} else {
		drawSea(screen) // [ФОН] море: вертикальний градієнт замість плоскої заливки

		// Тайли рівня. Стіни — темно-сірі, підлога не малюється (фон і є підлогою).
		//
		// [ПАКЕТУВАННЯ] Усі тайли — В ОДИН ШЛЯХ, а не по два виклики на кожен.
		//
		// Та сама техніка, що вже стоїть у ворсі, але виграш більший і причина інша:
		// стін у кадрі 218, тобто 436 викликів vector щокадру, і ця цифра залежить від
		// РОЗМІРУ СВІТУ, а не від кількості юнітів. Саме вона мовчки підняла підлогу,
		// коли карта виросла вчетверо, — а юніти вже перетнули межу кадру.
		//
		// Чому це взагалі вирішує проблему: замір показав, що обчислень тут немає —
		// логіка тіку 1.16 мс і тесселяція 0.88 мс із бюджету 8.33 мс, тобто чверть.
		// Решту зʼїдало САМЕ ЧИСЛО ВИКЛИКІВ. А з vsync промах навіть на десяту
		// мілісекунди коштує рівно половини кадрів: 120 → 60, без проміжних значень.
		//
		// Малюнок не змінюється: кожен тайл лишається окремим замкненим підшляхом, тож
		// заливка й контур виходять ті самі, до пікселя.
		wallPath.Reset() // пакетна змінна: інакше алокація шляху щокадру
		buildWallPath(&wallPath)
		fillOp, edgeOp := pathOpts(wallFill), pathOpts(wallEdge)
		vector.FillPath(screen, &wallPath, nil, &fillOp)
		vector.StrokePath(screen, &wallPath, &vector.StrokeOptions{Width: cam.s(1)}, &edgeOp)

		// [МЕЖА РІВНЯ] Рамки тут НЕМА, і це свідомо. Камера відсікає центр огляду до
		// меж світу (camera.go, follow), тож край карти ЗАВЖДИ лежить рівно на краю
		// вікна — при будь-якому зумі. Рамка збіглася б із ним піксель у піксель:
		// половина обводки за екраном, половина дублює віконну раму.
		//
		// Раніше тут стояв StrokeRect на screenWidth×screenHeight. Поки світ дорівнював
		// вікну, він лежав саме на краю екрана й читався як оформлення. Коли світ виріс
		// удвічі, той самий прямокутник лишився на СТАРИХ межах — і став фальшивою
		// стіною посеред карти: колір стіни (wallEdge), а юніти проходять крізь неї.

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
					vector.FillRect(screen, cam.px(x), cam.py(y), cam.s(pixelSize), cam.s(pixelSize),
						color.RGBA{255, 80, 0, uint8(a)}, false)
				}
			}
		}

		// [FLOW-FIELD] Поле напрямків до гравця (клавіша V) — під юнітами,
		// щоб стрілки не перекривали ворогів і гравця.
		if showFlowField != 0 {
			g.drawFlowField(screen)
		}

		// ══════════════════════════════════════════════════════════════════
		// [ДВА ПРОХОДИ] Спершу ШЛЯХИ всіх юнітів, потім ПРОСТІ ФІГУРИ всіх.
		//
		// Раніше кожен юніт малювався цілком: ворс(шлях) → долоні(коло) → тіло(шлях) →
		// кінчик(коло) → кульки(кола) → смуга HP(прямокутники). А ці два механізми в
		// ebiten різні (див. [ПАКЕТУВАННЯ] вище): шляхи НАКОПИЧУЮТЬСЯ й ідуть одним
		// викликом, а кожна проста фігура малюється негайно — і тим САМИМ скидає
		// накопичене. Дев'ять фігур на юніта давали 450 розривів пакета за кадр.
		//
		// Розділені проходи — ОДИН розрив: усі шляхи збираються разом, потім усі
		// фігури. Механізм при цьому не міняється: ми вже пробували перевести фігури
		// на шляхи й отримали вдвічі гірший кадр (30 FPS проти 63), бо шлях
		// растеризується через атлас, а коло — це просто трикутники.
		//
		// ЦІНА, і вона видима: юніт більше не малюється «цілком». Раніше пізніший юніт
		// перекривав ранішнього повністю; тепер тіла всіх лежать під кульками всіх.
		// У щільній купі це помітно — і це свідомий обмін на кадр.
		// ══════════════════════════════════════════════════════════════════

		// Діагностичні оверлеї — окремо й ПЕРШИМИ, щоб лишитись під юнітами. Вони
		// негайні, але їх один прохід, а не вперемішку з кожним юнітом.
		if showDetectionCircle || showWhiskers {
			for _, e := range g.units {
				cx := e.X + pixelSize/2
				cy := e.Y + pixelSize/2
				if showDetectionCircle {
					vector.StrokeCircle(screen, cam.px(cx), cam.py(cy), cam.s(e.Cfg.DetectionRange), cam.s(1),
						color.RGBA{255, 255, 255, 5}, false)
				}
				if showWhiskers && e.Brain != nil {
					drawBrainSensors(screen, e, cx, cy)
				}
			}
		}

		// Прохід 1 — ШЛЯХИ. Порядок усередині юніта збережено: ворс і кінцівки під
		// тілом, відросток над ним.
		for i := range g.units {
			if !cam.visible(g.units[i].X, g.units[i].Y) {
				continue // [ЗУМ] за кадром: при 4× це 15/16 світу
			}
			drawFur(screen, &g.units[i])
			drawLimbStrokes(screen, &g.units[i])
			drawPixelBody(screen, g.units[i])
			drawTentacleStroke(screen, &g.units[i])
		}
		drawFur(screen, &g.player)
		drawLimbStrokes(screen, &g.player)
		drawPixelBody(screen, g.player)
		drawTentacleStroke(screen, &g.player)

		// Прохід 2 — ПРОСТІ ФІГУРИ. Той самий відносний порядок, що був усередині
		// юніта: долоні, кінчик відростка, кульки, смуга HP.
		for i := range g.units {
			if !cam.visible(g.units[i].X, g.units[i].Y) {
				continue
			}
			drawLimbTips(screen, &g.units[i])
			drawTentacleTip(screen, &g.units[i])
			drawBalls(screen, &g.units[i])
			drawPixelOverlay(screen, g.units[i])
		}
		drawLimbTips(screen, &g.player)
		drawTentacleTip(screen, &g.player)
		drawBalls(screen, &g.player)
		drawPixelOverlay(screen, g.player)

		drawDash(screen, &g.player)
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
	// [ТЕМП] Цільовий TPS поруч із фактичним: коли вони розходяться, це просадка, а коли
	// цільовий не той, що очікуєш, — ти просто забув, що перемкнув темп клавішею T.
	if gameTPS != 120 {
		drawText(screen, fmt.Sprintf("temp %d", gameTPS), 10, screenWidth/2+90, 10,
			color.RGBA{240, 200, 90, 255})
	}
	// [КАМЕРА] Те саме правило, що для темпу: показуємо, лише коли зум НЕ типовий.
	// Інакше при 200% нічого не пояснює, чому поле раптом вужче — а це не баг, а
	// заплачена ціна за наближення.
	if cam.zoom != camZoomMin {
		drawText(screen, fmt.Sprintf("zoom %.0f%%", cam.zoom*100), 10, screenWidth/2+90, 25,
			color.RGBA{240, 200, 90, 255})
	}
	// [РЕНДЕР] Те саме правило, що для темпу й зуму: показуємо лише НЕтиповий стан.
	// Тут воно ще й обовʼязкове: якщо міряєш FPS із вимкненим згладжуванням і не
	// бачиш цього на екрані, наступного дня матимеш замір без відомої умови.
	if !antiAlias {
		drawText(screen, "AA off", 10, screenWidth/2+170, 10, color.RGBA{240, 200, 90, 255})
	}

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
