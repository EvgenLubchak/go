package main

import (
	"fmt"
	"image"
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
// faceInk — чорнило обличчя: очі, а далі й рот. Одне місце, щоб вони не розʼїхались.
var faceInk = color.RGBA{20, 20, 30, 255}

// toothWhite — трохи тепліший за чистий білий: крижаний білок серед мʼяких тіл
// виглядав би стороннім елементом, а не частиною істоти.
var toothWhite = color.RGBA{245, 243, 235, 255}

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
// перемішувати порядок — саме це й роблять три проходи в Draw ([ДВА ПРОХОДИ] нижче).
//
// Прапорець згладжування при цьому один на весь рендер: перемикання antialias теж
// скидає батч, тож змішувати режими в одному кадрі — та сама вада.
// ══════════════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════════════
// [ЗГЛАДЖУВАННЯ] ЩО НАСПРАВДІ КОШТУЄ ДОРОГО (vector/fill.go у ebiten 2.9)
//
// Спершу я пояснив це офскрін-буфером із internal/ui/image.go і написав pinAARegion,
// щоб прибити його рамку. Пояснення ХИБНЕ: той буфер вмикається лише від
// DrawTrianglesOptions.AntiAlias, а пакет vector його НЕ виставляє ніде. Приріст
// 10–15%, який ми тоді побачили, дав градієнт моря з того ж коміту — я поміряв два
// важелі разом і приписав ефект не тому.
//
// Справжній механізм інший:
//
//	1. КОЖЕН шлях малюється у трафарет ВІСІМ разів із субпіксельними зсувами
//	   (offsetAndColorsAA проти одного зсуву без AA) — тобто ×8 викликів;
//	2. кожен шлях отримує свою область в атласі = його ГАБАРИТИ, округлені до 16px,
//	   і при AA ця область ВДВІЧІ ШИРША;
//	3. атлас чиститься щокадру, а що не влізло в 4093×4093 — це ще одне зображення.
//
// Тобто ціна AA росте з КІЛЬКІСТЮ ШЛЯХІВ у кадрі, а не з площею малюнка. У нас 10
// шляхів на юніта, тож 50 видимих юнітів це 500 шляхів → 4000 малювань у трафарет
// проти 500. Саме тому просадка приходить у бою: юніти збігаються в купу й разом
// потрапляють у кадр, а поза боєм вони розмазані по світу, що вчетверо більший за вікно.
//
// pinAARegion був ще й активно шкідливим: шлях на весь екран отримував область
// 3424×992 з 4093 доступних, і атлас роздувався з 1.2 до 16.5 Мпx, вимагаючи ДРУГЕ
// зображення. Видалено.
//
// ЗВІДСИ РІШЕННЯ — beginScene/endScene нижче: згладжувати не кожен шлях окремо, а
// весь кадр одразу.
// ══════════════════════════════════════════════════════════════════════════

// pathOpts — спільні опції малювання шляху. Єдине місце, де читається antiAlias:
// якби кожна функція вирішувала сама, вони б розʼїхались і батч рвався б знову.
func pathOpts(col color.RGBA) vector.DrawPathOptions {
	var op vector.DrawPathOptions
	op.AntiAlias = antiAlias
	op.ColorScale.ScaleWithColor(col)
	return op
}

// sceneBuf — збільшений буфер, у який малюється світ при renderScale > 1.
//
// Пакетна змінна: виділяється раз і живе. У цьому й уся суть на тлі AA — там атлас
// перебудовувався щокадру під поточні шляхи, а тут одна текстура назавжди.
var sceneBuf *ebiten.Image

// ssLadder — щаблі суперсемплінгу, які циклює панель (Tab → «Графіка»).
//
// Список, а не крок: між 1.5 і 2 різниця тонка, а між 2 і 4 — прірва, тож рівномірна
// сітка була б або надто дрібною внизу, або надто грубою вгорі.
//
// [ЧОМУ ДРАБИНА ЗАКІНЧУЄТЬСЯ НА ×4] Спершу тут стояли ще ×8 і ×12. На ×12 гра
// ВПАЛА — і впала так, що newSceneBuf нижче цього не перехопив.
//
// Це головний урок цього місця, і він коштував краху: буфер ×12 від вікна 1700×980 це
// 20400×11760 пікселів, майже гігабайт, і невдача такого виділення НЕ приходить як
// Go-паніка. Вона стається глибше — у драйвері або на самому виділенні памʼяті, куди
// recover не дістає взагалі. Мій захист був теоретичним: він ловить лише той клас
// відмов, який ebiten оформлює як panic, а справжня стеля заліза лежить нижче.
//
// Звідси правило: у драбині лишаються ЛИШЕ перевірені в грі щаблі. Верхню межу тут
// визначає не наша сміливість, а те, що справді запустилось.
//
// ×8 прибрано теж — 427 МБ і 64 пікселі на кожен екранний заради картинки, яка вже не
// кращає (див. про 2×2 тексели в endScene).
var ssLadder = []float32{1, 1.5, 2, 4}

// newSceneBuf пробує виділити буфер і чесно каже, чи вийшло.
//
// [GO: RECOVER] ebiten.NewImage на неприйнятному розмірі ПАНІКУЄ, і цю паніку ми тут
// ловимо. Але захист ЧАСТКОВИЙ, і це перевірено крахом: на ×12 гра впала повз recover,
// бо відмова виділення такого розміру приходить не з Go, а з драйвера.
//
// Тобто ця функція страхує від дрібної помилки в арифметиці розміру, а не від виходу
// за стелю заліза. Від стелі страхує ssLadder, у якому лишаються тільки перевірені
// щаблі.
func newSceneBuf(scale float32) (img *ebiten.Image) {
	defer func() {
		if recover() != nil {
			img = nil
		}
	}()
	return ebiten.NewImage(int(screenWidth*scale), int(screenHeight*scale))
}

// beginScene повертає зображення, у яке малювати СВІТ.
//
// [СУПЕРСЕМПЛІНГ] renderScale > 1 → малюємо в буфер більшого розміру, а потім
// стискаємо його на екран із лінійною фільтрацією. Чотири пікселі буфера дають один
// піксель екрана, тобто край, що проходить через піксель навскіс, отримує проміжний
// відтінок — те саме, чого ми хотіли від AA.
//
// ЧОМУ ЦЕ КРАЩЕ ЗА AA САМЕ В НАШОМУ ВИПАДКУ. Ціна per-path AA росте з кількістю
// ШЛЯХІВ (див. блок вище): у бою юніти збігаються в кадр, шляхів стає 500, малювань —
// 4000, і кадр валиться. Ціна суперсемплінгу — СТАЛА: одна текстура й одне стискання,
// байдуже, скільки на екрані юнітів і чи вони б'ються. Ми міняємо змінну ціну на
// фіксовану, і саме нестабільність була проблемою, а не середнє значення.
//
// Побічно: згладжується ВСЕ однаково — ворс, жало, тіло, стіни, — а не лише те, що
// намальоване шляхом. Кульки з долонями досі йшли повз AA, бо малюються трикутниками.
//
// [СПУСК ДРАБИНОЮ] Якщо запитаний масштаб не влазить у текстуру, пробуємо щабель нижче,
// і так до 1. Тобто верхні щаблі можна вмикати без ризику: гра або покаже їх, або
// чесно скаже на HUD, що дісталась не туди, куди просили.
func beginScene() *ebiten.Image {
	if renderScale <= 1 {
		sceneScale = 1
		return screenImage
	}
	w := int(screenWidth * renderScale)
	h := int(screenHeight * renderScale)
	if sceneBuf != nil && sceneBuf.Bounds().Dx() == w && sceneBuf.Bounds().Dy() == h {
		return sceneBuf // той самий масштаб, що й торік — буфер живе далі
	}
	// Звільняємо СПЕРШУ: на верхніх щаблях буфер важить сотні мегабайтів, і тримати
	// старий разом із новим означало б впертись у памʼять там, де вистачило б однієї.
	if sceneBuf != nil {
		sceneBuf.Deallocate()
		sceneBuf = nil
	}
	for i := len(ssLadder) - 1; i >= 0; i-- {
		s := ssLadder[i]
		if s > renderScale || s <= 1 {
			continue
		}
		if buf := newSceneBuf(s); buf != nil {
			sceneBuf, sceneScale = buf, s
			return sceneBuf
		}
	}
	sceneScale = 1
	return screenImage
}

// endScene стискає буфер на екран. При sceneScale = 1 світ уже намальований на екрані.
//
// Фільтр лінійний, тобто читає 2×2 тексели. На ×2 це рівно ті чотири пікселі, з яких
// складається екранний, — ідеальне усереднення. На ×4 таких пікселів уже 16, а
// прочитає він знову чотири, тож ebiten підмішує мипмапи. Звідси й чесна межа
// корисності: вище ×2 картинка майже не кращає, а платня росте як КВАДРАТ множника.
func endScene(screen *ebiten.Image) {
	if sceneScale <= 1 {
		return
	}
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
	op.GeoM.Scale(1/float64(sceneScale), 1/float64(sceneScale))
	screen.DrawImage(sceneBuf, op)
}

// screenImage — екран поточного кадру. Потрібен лише beginScene при renderScale = 1,
// щоб не тягнути екран параметром крізь виклик, який його здебільшого не використовує.
var screenImage *ebiten.Image

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

// [ГРАДІЄНТ] Біла точка як джерело кольору для DrawTriangles.
//
// GPU множить колір текстури на колір ВЕРШИНИ, тож біле джерело означає «бери колір
// цілком із вершини». Зображення 3×3 із вирізаною серединою — щоб білінійна фільтрація
// не зачепила краю й не підмішала прозорість: той самий прийом, що в самого ebiten.
var (
	whitePix = ebiten.NewImage(3, 3)
	whiteDot = whitePix.SubImage(image.Rect(1, 1, 2, 2)).(*ebiten.Image)
)

func init() {
	pix := make([]byte, 4*3*3)
	for i := range pix {
		pix[i] = 0xff
	}
	whitePix.WritePixels(pix)
}

// seaColorAt — колір моря на заданій СВІТОВІЙ висоті. 0 = поверхня, worldHeight = дно.
//
// Окремою функцією заради тесту: сам градієнт малює GPU, і назовні він нічого не
// віддає — перевірити, що глибина не перевернулась, можна лише тут.
func seaColorAt(worldY float32) color.RGBA {
	t := worldY / worldHeight
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return color.RGBA{
		R: uint8(float32(seaTopR) + (seaBotR-seaTopR)*t),
		G: uint8(float32(seaTopG) + (seaBotG-seaTopG)*t),
		B: uint8(float32(seaTopB) + (seaBotB-seaTopB)*t),
		A: 255,
	}
}

// worldYAt — обернення cam.py: яка СВІТОВА висота лежить на цьому рядку екрана.
func worldYAt(screenY float32) float32 {
	// Ділимо на sceneScale: cam.py на нього множить, тож обернення мусить симетрично
	// скасувати масштаб — інакше при суперсемплінгу градієнт розтягнеться вдвічі.
	return (screenY/sceneScale-screenHeight/2)/cam.zoom + cam.cy
}

// drawSea малює фон-море: вертикальний градієнт від світлішої мілини вгорі до темної
// глибини внизу.
//
// Чому градієнт, а не screen.Fill одним кольором: плоска заливка читається як
// «пофарбоване тло», а вертикальний перехід — як ГЛИБИНА, і цього досить, щоб поле
// перестало бути абстрактним аркушем.
//
// [КАМЕРА] Колір прив'язаний до СВІТОВОГО Y, а не до екранного. Поки світ дорівнював
// вікну, різниці не було. Коли світ став удвічі вищим, екранна прив'язка перетворила
// глибину на скайбокс: та сама світова клітинка світлішала й темнішала залежно від
// того, де стоїть камера. Зі світовою — рухаючись вертикально, ти справді занурюєшся.
//
// [ОДИН ЧОТИРИКУТНИК ЗАМІСТЬ 48 СМУГ]
//
// Раніше тут був цикл на seaBands прямокутників: градієнт СХОДИНКАМИ, бо FillRect
// заливає одним кольором. Тепер це чотири вершини, у яких колір заданий згори й знизу,
// а проміжок інтерполює GPU — саме те, що він робить апаратно й задарма.
//
// Виграш подвійний, і другий важливіший за перший:
//
//	48 викликів → 1;
//	сходинки зникли зовсім — перехід став неперервним, а не «майже».
//
// Разом із ними зникла й ручка seaBands, і кострубатий +1 піксель до висоти смуги:
// він затуляв волосяні щілини, що народжувались при округленні між смугами. Немає
// смуг — немає щілин. Класичний випадок, коли зникає не лише код, а й проблема.
func drawSea(screen *ebiten.Image) {
	// Кольори беремо на верхньому й нижньому краях ВИДИМОЇ частини світу, а не всієї
	// карти: інакше при зумі градієнт розтягнувся б на екран цілком і глибина знову
	// поїхала б за камерою.
	b := screen.Bounds()
	w, h := float32(b.Dx()), float32(b.Dy())
	top := seaColorAt(worldYAt(0))
	bot := seaColorAt(worldYAt(h))

	v := func(x, y float32, c color.RGBA) ebiten.Vertex {
		return ebiten.Vertex{
			DstX: x, DstY: y,
			SrcX: 1, SrcY: 1, // середина білої точки
			ColorR: float32(c.R) / 255,
			ColorG: float32(c.G) / 255,
			ColorB: float32(c.B) / 255,
			ColorA: 1,
		}
	}
	verts := [4]ebiten.Vertex{
		v(0, 0, top), v(w, 0, top),
		v(0, h, bot), v(w, h, bot),
	}
	op := &ebiten.DrawTrianglesOptions{ColorScaleMode: ebiten.ColorScaleModePremultipliedAlpha}
	screen.DrawTriangles(verts[:], []uint16{0, 1, 2, 1, 3, 2}, whiteDot, op)
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

// strikeColorOf — колір відростка й кульок: білий, поки триває спалах удару.
//
// [ДИЗАЙН: ДВА СПАЛАХИ НА РІЗНИХ ЧАСТИНАХ ТІЛА] Біле тіло вже означає «мене вдарили»
// (HitTimer у drawPixelBody). Новий спалах означає протилежне — «вдарив я». Якби він
// теж світив тіло, дві події стали б НЕРОЗРІЗНЕННИМИ, і саме там, де це найпотрібніше:
// у щільній бійці ти бачив би білі спалахи й не знав, хто кому завдав.
//
// Тому нападник світить ВІДРОСТКОМ І КУЛЬКАМИ. Читається як розряд, що виходить крізь
// кінцівку, якою його й доставили: жало в цю саму мить вистрілює вперед, тож рух і
// колір розповідають одну подію двома каналами.
//
// Одним помічником, а не трьома однаковими if: три місця розʼїхались би, і відросток
// світився б без кульок.
func strikeColorOf(p *Pixel) color.RGBA {
	if p.StrikeTimer > 0 {
		return color.RGBA{255, 255, 255, 255}
	}
	return p.Color
}

// drawBalls — [КУЛЬКИ] дві кульки, що бовтаються під тілом.
//
// Малюємо ПІСЛЯ тіла, а не до: вони мають лишатись видимими цілком, коли підтягуються
// під корпус. Ворс навпаки йде до тіла, щоб корені ховались — різні цілі, різний порядок.
//
// Колір — тіла, без прозорості: кульки читаються як частина істоти, а не як ефект.
func drawBalls(screen *ebiten.Image, p *Pixel) {
	for i := 0; i < ballCount; i++ {
		vector.FillCircle(screen, cam.px(p.Balls[i][0]), cam.py(p.Balls[i][1]), cam.s(ballRadius), strikeColorOf(p), antiAlias)
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
	op := pathOpts(strikeColorOf(p))
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
	vector.FillCircle(screen, cam.px(px), cam.py(py), cam.s(tentTipBall), strikeColorOf(p), false)
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

// drawEyes — [ОБЛИЧЧЯ] дві крапки-ока й рот. Другий прохід: це прості фігури,
// тобто ті самі трикутники, що смуги HP, і власного розриву пакета вони не дають.
//
// ЧОМУ НЕ ШРИФТ. Раніше обличчя малювала мітка (`*_*`) через drawText — а це АТЛАС
// ГЛІФІВ, інше джерельне зображення, ніж усі наші трикутники. Кожна мітка давала два
// розриви пакета: у текст і назад. На тисячах юнітів це обвалювало FPS — зміряно ще
// до цього проєкту, і саме тому мітки роками тримали лише на десятках юнітів.
//
// Зіниця несе СПРИЙНЯТТЯ: куди агент дивиться і чи бачить ціль узагалі. Тіло вже
// показує НАМІР (форма з Q), тож обличчя навмисно взяло інший канал — інакше вийшла
// б друга копія того самого приладу.
//
// Побічний наслідок, вартий окремої уваги: localSight досі був НЕВИДИМИЙ. Ти ховався
// за стіною й лише здогадувався, що тебе загубили. Тепер це видно очима.
func drawEyes(screen *ebiten.Image, p *Pixel) {
	cx := p.X + pixelSize/2
	cy := p.Y + pixelSize/2
	scale := bodyScaleOf(p)
	for i := 0; i < 2; i++ {
		ex, ey := eyeRoot(i, cx, cy, scale)
		// Зсув погляду — прямо в позицію ока: окремого білка немає, тож саме око і є
		// зіницею. На різнокольорових тілах це єдиний варіант, що читається завжди.
		ex += p.Pupil[0] * float32(pupilShift) * scale
		ey += p.Pupil[1] * float32(pupilShift) * scale
		vector.FillCircle(screen, cam.px(ex), cam.py(ey), cam.s(float32(eyeRadius)*scale),
			faceInk, antiAlias)
	}

	// [РОТ] Капсула: прямокутник посередині й коло на кожному кінці.
	//
	// Саме цими примітивами, а не шляхом, з двох причин. Шлях малюється ПЕРШИМ
	// проходом і опинився б ПІД тілом. А довільний багатокутник через DrawTriangles
	// вимагав би НАШОГО білого зображення — іншого джерела, ніж у vector, — і дав би
	// пінг-понг пакетів упереміш з очима. FillRect і FillCircle же йдуть з одного
	// джерела, тож уся капсула лягає в той самий пакет, що й очі.
	w, h, r := mouthShape(p.Mouth, scale)
	my := cy + float32(mouthOffsetY)*scale
	if inner := w - 2*r; inner > 0 {
		vector.FillRect(screen, cam.px(cx-inner/2), cam.py(my-h/2),
			cam.s(inner), cam.s(h), faceInk, antiAlias)
	}
	vector.FillCircle(screen, cam.px(cx-w/2+r), cam.py(my), cam.s(r), faceInk, antiAlias)
	vector.FillCircle(screen, cam.px(cx+w/2-r), cam.py(my), cam.s(r), faceInk, antiAlias)

	appendTeeth(p, cx, my, w, h, r)
}

// [ЗУБИ] Спільний буфер вершин на весь кадр. Пакетні змінні, а не локальні: інакше
// кожен кадр алокував би слайси на сотні трикутників і віддавав їх збирачу сміття.
var (
	teethVerts []ebiten.Vertex
	teethIdx   []uint16
)

// addTooth кладе один трикутник у спільний буфер. Координати — СВІТОВІ; у екранні
// переводимо тут, бо далі вершини вже нікуди не рухаються.
func addTooth(x0, y0, x1, y1, x2, y2 float32) {
	base := uint16(len(teethVerts))
	v := func(x, y float32) ebiten.Vertex {
		return ebiten.Vertex{
			DstX: cam.px(x), DstY: cam.py(y),
			SrcX: 1, SrcY: 1, // середина білої точки
			ColorR: float32(toothWhite.R) / 255,
			ColorG: float32(toothWhite.G) / 255,
			ColorB: float32(toothWhite.B) / 255,
			ColorA: 1,
		}
	}
	teethVerts = append(teethVerts, v(x0, y0), v(x1, y1), v(x2, y2))
	teethIdx = append(teethIdx, base, base+1, base+2)
}

// flushTeeth віддає ВСІ зуби кадру одним викликом і чистить буфер.
func flushTeeth(screen *ebiten.Image) {
	if len(teethIdx) == 0 {
		return
	}
	op := &ebiten.DrawTrianglesOptions{ColorScaleMode: ebiten.ColorScaleModePremultipliedAlpha}
	screen.DrawTriangles(teethVerts, teethIdx, whiteDot, op)
	teethVerts, teethIdx = teethVerts[:0], teethIdx[:0]
}

// mouthHalfHeightAt — піввисота капсули на горизонтальному зсуві dx від центра рота.
//
// Потрібно, бо коли рот найбільш розкритий, він майже круглий і РІВНОЇ кромки в нього
// майже немає: inner = w − 2r сходиться до нуля. Зуби, посаджені на пряму лінію,
// вилізли б за контур на обличчя.
//
// Тому основа кожного зуба сідає на САМ контур: на пласкій ділянці це h/2, на
// заокругленнях — коло радіуса r. Виходить щелепа, а не наліпка.
func mouthHalfHeightAt(dx, w, h, r float32) float32 {
	flat := w/2 - r // піввисота стала, доки не почалось заокруглення
	if dx < 0 {
		dx = -dx
	}
	if dx <= flat {
		return h / 2
	}
	d := dx - flat
	if d >= r {
		return 0
	}
	return float32(math.Sqrt(float64(r*r - d*d)))
}

// appendTeeth складає зуби юніта у СПІЛЬНИЙ буфер вершин.
//
// [ОДИН ВИКЛИК НА ВЕСЬ КАДР] Трикутника в пакеті vector немає, тож малюємо через
// DrawTriangles із власним білим зображенням — так само, як море. Це інше джерело, ніж
// у vector, тобто розрив пакета. Але розрив буде ОДИН на кадр, а не на юніта: усі зуби
// всіх юнітів накопичуються тут і віддаються разом (flushTeeth).
//
// Виходить дешевше за очі: 50 юнітів × 6 зубів це 300 трикутників одним викликом,
// тоді як очі дають по два виклики на юніта.
//
// Відсікання не потрібне: рот у нас не ДІРА, а чорна фігура поверх тіла, тож зуби —
// просто білі трикутники поверх чорного.
func appendTeeth(p *Pixel, cx, my, w, h, r float32) {
	if toothCount <= 0 || p.Mouth < toothMinOpen {
		return
	}
	// Зуби наростають від нуля на межі появи — інакше вони вискакували б цілими.
	grow := (p.Mouth - float32(toothMinOpen)) / (1 - float32(toothMinOpen))

	for i := 0; i < toothCount; i++ {
		// Рівномірно по ширині, з півкроком від країв: інакше крайні зуби сиділи б
		// рівно на кінчиках капсули, де висоти вже немає.
		t := (float32(i) + 0.5) / float32(toothCount)
		dx := (t - 0.5) * w
		half := mouthHalfHeightAt(dx, w, h, r)
		if half <= 0 {
			continue
		}
		tipLen := half * float32(toothDepth) * grow
		base := w / float32(toothCount) / 2 // піврозмах основи

		for _, dir := range [2]float32{-1, +1} { // верхня щелепа й нижня
			y0 := my + dir*half // основа НА контурі
			addTooth(
				cx+dx-base, y0,
				cx+dx+base, y0,
				cx+dx, y0-dir*tipLen, // вістря всередину рота
			)
		}
	}
}

// mouthShape — ширина, висота й радіус кінців за станом рота (0 = риска, 1 = «о»).
//
// Окремою функцією заради тесту: малювання йде прямо в ebiten і назовні не віддає
// нічого, а перевіряти тут є що — головна властивість рота саме в тому, що при
// розкритті змінюється ФОРМА, а не лише розмір.
func mouthShape(open, scale float32) (w, h, r float32) {
	lerp := func(a, b float64) float32 { return float32(a + (b-a)*float64(open)) }
	w = lerp(mouthLineW, mouthRoundW) * scale
	h = lerp(mouthLineH, mouthRoundH) * scale
	r = h / 2
	if r > w/2 {
		r = w / 2 // кінці не можуть бути товщі за саму фігуру
	}
	return w, h, r
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
// drawWorld малює СВІТ (усе, що живе у світових координатах) у задане зображення.
//
// Окремою функцією, щоб її можна було націлити не на екран, а на збільшений буфер —
// див. beginScene. Параметр названий screen навмисно: тіло писалось під екран, і
// перейменування нічого б не додало, крім шуму в дифі.

// Draw малює поточний стан на екрані.
// drawWorld малює СВІТ (усе, що живе у світових координатах) у задане зображення.
//
// Окремою функцією, щоб її можна було націлити не на екран, а на збільшений буфер —
// див. beginScene. Параметр названий screen навмисно: тіло писалось під екран, і
// перейменування нічого б не додало, крім шуму в дифі.
func (g *Game) drawWorld(screen *ebiten.Image) {
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

	// Прохід 2 — ПРОСТІ ФІГУРИ, усі негайні: долоні, кінчик відростка, смуга HP.
	for i := range g.units {
		if !cam.visible(g.units[i].X, g.units[i].Y) {
			continue
		}
		drawLimbTips(screen, &g.units[i])
		drawTentacleTip(screen, &g.units[i])
		drawEyes(screen, &g.units[i])
		drawPixelOverlay(screen, g.units[i])
	}
	drawLimbTips(screen, &g.player)
	drawTentacleTip(screen, &g.player)
	drawEyes(screen, &g.player)
	drawPixelOverlay(screen, g.player)
	// [ЗУБИ] Один виклик на всі обличчя кадру — саме заради цього вони й копились.
	flushTeeth(screen)

	// Прохід 3 — КУЛЬКИ, останніми й окремо. Вони єдина проста фігура, що йде за
	// перемикачем згладжування, і саме тому не можуть стояти всередині проходу 2:
	// AA-кулька малюється через шлях, а сусідні долоні й смуга HP — негайно, тож
	// вони скидали б її пакет НА КОЖНОМУ юніті. Винесені в кінець — усі кульки
	// кадру збираються в один пакет, і зі згладжуванням це один буфер, а не 50.
	//
	// Порядок від цього не страждає: кульки й так були поверх усього, а зі смугою
	// HP вони не перетинаються — та висить над юнітом, ці бовтаються під ним.
	for i := range g.units {
		if !cam.visible(g.units[i].X, g.units[i].Y) {
			continue
		}
		drawBalls(screen, &g.units[i])
	}
	drawBalls(screen, &g.player)

	drawDash(screen, &g.player)
}

// Draw малює поточний стан на екрані.
func (g *Game) Draw(screen *ebiten.Image) {
	screenImage = screen
	if g.firstPerson {
		// [RAYCASTER] Вид від першої особи замість топ-дауну (клавіша F).
		g.drawFirstPerson(screen)
	} else {
		g.drawWorld(beginScene())
		endScene(screen)
	}

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
	if gameTPS != gameTPSDefault {
		drawText(screen, fmt.Sprintf("temp %d", gameTPS), 10, screenWidth/2+90, 10,
			color.RGBA{240, 200, 90, 255})
	}
	// [КАМЕРА] Те саме правило, що для темпу: показуємо, лише коли зум НЕ типовий.
	// Інакше при 200% нічого не пояснює, чому поле раптом вужче — а це не баг, а
	// заплачена ціна за наближення.
	if cam.zoom != camZoomDefault {
		drawText(screen, fmt.Sprintf("zoom %.0f%%", cam.zoom*100), 10, screenWidth/2+90, 25,
			color.RGBA{240, 200, 90, 255})
	}
	// [РЕНДЕР] Те саме правило, що для темпу й зуму: показуємо лише НЕтиповий стан.
	// Тут воно ще й обовʼязкове: якщо міряєш FPS із вимкненим згладжуванням і не
	// бачиш цього на екрані, наступного дня матимеш замір без відомої умови.
	if antiAlias != antiAliasDefault {
		// Напис описує ФАКТИЧНИЙ стан, а умова — відхилення від типового. Якби напис був
		// прибитий до «AA off», зміна дефолту зробила б його брехнею мовчки.
		label := "AA off"
		if antiAlias {
			label = "AA on"
		}
		drawText(screen, label, 10, screenWidth/2+170, 10, color.RGBA{240, 200, 90, 255})
	}
	if renderScale != renderScaleDefault || sceneScale != renderScale {
		// Розбіжність показуємо ЗАВЖДИ, навіть на типовому масштабі: «просив 12, дали 8»
		// це не дрібниця, а межа заліза, і мовчати про неї означало б збрехати про те,
		// що зараз на екрані.
		label := fmt.Sprintf("SS %.2g×", sceneScale)
		if sceneScale != renderScale {
			label = fmt.Sprintf("SS %.2g×→%.2g×", renderScale, sceneScale)
		}
		drawText(screen, label, 10, screenWidth/2+170, 25, color.RGBA{240, 200, 90, 255})
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

	// [ПАНЕЛЬ НАЛАШТУВАНЬ] (Tab) — найвищий шар: модалка мусить лежати над HUD і
	// метриками, інакше клікабельний рядок може виявитись під чужим текстом.
	if panelOpen {
		drawPanel(screen)
	}
}
