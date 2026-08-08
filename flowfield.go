package main

import (
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// ==========================================================================
// FLOW-FIELD («поле потоку») — ПОШУК ШЛЯХУ, протилежність реактивності.
//
// Рій-учень реактивний: бачить гравця (або пам'ятає) і тисне в його бік — тож
// об стіну він таки б'ється, бо НЕ ЗНАЄ лабіринту. Flow-field вирішує це інакше:
//
//	1. Пускаємо BFS-хвилю ВІД ГРАВЦЯ по всіх прохідних клітинках.
//	   Кожна клітинка запам'ятовує, за скільки кроків до гравця (dist).
//	2. Напрямок у клітинці = у бік сусіда з МЕНШОЮ відстанню (градієнт вниз).
//
// Результат: у КОЖНІЙ точці карти є стрілка «туди до гравця крізь усі стіни».
// Будується раз (дешево: 67×38 ≈ 2500 клітинок), читається будь-якою кількістю
// агентів безкоштовно — тому це класика ігрового ШІ для сотень юнітів.
//
// [ВАЖЛИВО] Це ІНЖЕНЕРНЕ знання, не вивчене. Далі ми віддамо цю стрілку мозку
// вбивці як ВХІД — щоб мережа не витрачала ємність на лабіринт, а вчила тактику
// (коли кинутись, коли відступити). Добрі фічі сильніші за більшу мережу.
//
// BFS по 4 сусідах (щоб відстань була чесною), а напрямок — по 8 (щоб рух виходив
// плавним, по діагоналях). Діагональ заборонена, якщо «зрізає» кут крізь стіну.
// ==========================================================================

const flowUnreachable = -1 // клітинку не досягти від гравця (замкнена кишеня)

// flowOff4 — 4 ортогональні сусіди для BFS-хвилі.
var flowOff4 = [4][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}

// flowOff8 — 8 сусідів для градієнта; порядок збігається з dirs8 (N, NE, E, ...).
var flowOff8 = [8][2]int{
	{0, -1}, {1, -1}, {1, 0}, {1, 1},
	{0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
}

// flowSource — клітинка-джерело хвилі.
type flowSource struct{ col, row int }

// FlowField — поле напрямків до НАЙБЛИЖЧОГО джерела для всієї карти.
type FlowField struct {
	dist [boidMapH][boidMapW]int32   // кроків до найближчого джерела; -1 = недосяжно
	dirX [boidMapH][boidMapW]float32 // одиничний напрямок «куди йти»
	dirY [boidMapH][boidMapW]float32

	queue   []int32 // черга BFS, перевикористовується (без алокацій щокадру)
	maxDist int32   // найдальша досяжна клітинка (для розфарбування)
	valid   bool

	// [КАМЕРА] Кеш малювання прибрано: тепер світ може бути більшим за екран, і
	// створювати текстуру worldWidth×worldHeight (100+ МБ) для дебаг-оверлея
	// марнотратно. Натомість drawFlowField малює лише ВИДИМІ клітинки щокадру
	// з камерою — каллінг компенсує вартість теселяції.
}

// rebuildFrom — зручний варіант для ОДНОГО джерела (використовують тести).
func (f *FlowField) rebuildFrom(col, row int) {
	f.rebuild([]flowSource{{col, row}})
}

// rebuild — будує поле від УСІХ джерел одразу (multi-source BFS).
//
// [ЧОМУ БАГАТО ДЖЕРЕЛ] Юніту потрібен шлях до НАЙБЛИЖЧОГО супротивника, а не до
// одного конкретного. Замість поля на кожну ціль кладемо в стартову чергу ВСІ
// цілі одразу з відстанню 0 — і хвиля сама «розділить» карту: кожна клітинка
// отримає напрямок до тієї цілі, що ближча ЛАБІРИНТОМ. Ціна та сама: один прохід
// по ~2500 клітинок.
func (f *FlowField) rebuild(sources []flowSource) {
	f.valid = false
	// (кеш прибрано — малюємо щокадру з камерою)
	f.maxDist = 0
	for r := 0; r < boidMapH; r++ {
		for c := 0; c < boidMapW; c++ {
			f.dist[r][c] = flowUnreachable
			f.dirX[r][c], f.dirY[r][c] = 0, 0
		}
	}
	// --- Крок 1: BFS-хвиля від УСІХ джерел (відстань у клітинках) ---
	q := f.queue[:0]
	for _, s := range sources {
		if isWallAt(s.col, s.row) || f.dist[s.row][s.col] == 0 {
			continue // джерело в стіні або вже додане
		}
		f.dist[s.row][s.col] = 0
		q = append(q, int32(s.row*boidMapW+s.col))
	}
	if len(q) == 0 {
		return // джерел немає (усіх убито) → поля нема
	}
	for head := 0; head < len(q); head++ {
		cell := q[head]
		c := int(cell) % boidMapW
		r := int(cell) / boidMapW
		nd := f.dist[r][c] + 1
		for _, o := range flowOff4 {
			nc, nr := c+o[0], r+o[1]
			if isWallAt(nc, nr) || f.dist[nr][nc] != flowUnreachable {
				continue // стіна або вже відвідано (BFS → перший раз = найкоротше)
			}
			f.dist[nr][nc] = nd
			if nd > f.maxDist {
				f.maxDist = nd
			}
			q = append(q, int32(nr*boidMapW+nc))
		}
	}
	f.queue = q // тримаємо ємність для наступних перебудов

	// --- Крок 2: напрямок = до сусіда з найменшою відстанню (спуск градієнтом) ---
	for r := 0; r < boidMapH; r++ {
		for c := 0; c < boidMapW; c++ {
			if f.dist[r][c] == flowUnreachable {
				continue
			}
			best := f.dist[r][c]
			var bx, by int
			for _, o := range flowOff8 {
				nc, nr := c+o[0], r+o[1]
				if isWallAt(nc, nr) {
					continue
				}
				// Діагональ дозволена лише якщо не зрізає кут крізь стіну.
				if o[0] != 0 && o[1] != 0 && (isWallAt(c+o[0], r) || isWallAt(c, r+o[1])) {
					continue
				}
				if d := f.dist[nr][nc]; d != flowUnreachable && d < best {
					best, bx, by = d, o[0], o[1]
				}
			}
			if bx != 0 || by != 0 {
				l := float32(math.Sqrt(float64(bx*bx + by*by)))
				f.dirX[r][c] = float32(bx) / l
				f.dirY[r][c] = float32(by) / l
			}
			// (клітинки-джерела лишаються (0,0) — ми вже на місці)
		}
	}
	f.valid = true
}

// dirAt — напрямок до найближчого джерела з точки (x,y). ok=false, якщо
// поле не готове або точка в недосяжній кишені. Знадобиться мозку вбивці.
func (f *FlowField) dirAt(x, y float32) (dx, dy float32, ok bool) {
	c := int(x+pixelSize/2) / pixelSize
	r := int(y+pixelSize/2) / pixelSize
	if !f.valid || c < 0 || c >= boidMapW || r < 0 || r >= boidMapH {
		return 0, 0, false
	}
	if f.dist[r][c] == flowUnreachable {
		return 0, 0, false
	}
	return f.dirX[r][c], f.dirY[r][c], true
}

// distAt — відстань у клітинках до найближчого джерела крізь лабіринт (−1 = недосяжно).
func (f *FlowField) distAt(x, y float32) int32 {
	c := int(x+pixelSize/2) / pixelSize
	r := int(y+pixelSize/2) / pixelSize
	if !f.valid || c < 0 || c >= boidMapW || r < 0 || r >= boidMapH {
		return flowUnreachable
	}
	return f.dist[r][c]
}

// cellOf — клітинка сітки, у якій стоїть юніт.
func cellOf(p *Pixel) flowSource {
	return flowSource{int(p.X+pixelSize/2) / pixelSize, int(p.Y+pixelSize/2) / pixelSize}
}

// updateFlowFields — [КОМАНДИ] перебудовує ОБИДВА поля:
//
//	flowToPlayerSide — джерела: гравець + його юніти → читають ВОРОГИ
//	flowToEnemySide  — джерела: ворожі юніти        → читають ЮНІТИ ГРАВЦЯ
//
// Симетрично: кожна сторона має маршрут до найближчого супротивника крізь стіни.
// Побічний (бажаний) ефект: вороги більше не пробігають повз твоїх юнітів до тебе
// — вони йдуть на найближчого з вашого боку, тож перехоплення реально працює.
//
// Троттлимо раз на flowRebuildEvery кадрів: джерел багато й вони весь час рухаються,
// тож «перебудова при зміні клітинки» вже не економила б. 2 поля × 2500 клітинок
// × 20 разів/с — мізер.
func (g *Game) updateFlowFields() {
	if g.flowTick > 0 && g.tick%flowRebuildEvery != 0 {
		return
	}
	g.flowTick++

	playerSide := []flowSource{cellOf(&g.player)}
	var enemySide []flowSource
	for i := range g.units {
		u := &g.units[i]
		if u.HP <= 0 {
			continue
		}
		if u.Faction == factionPlayer {
			playerSide = append(playerSide, cellOf(u))
		} else {
			enemySide = append(enemySide, cellOf(u))
		}
	}
	g.flowToPlayerSide.rebuild(playerSide)
	g.flowToEnemySide.rebuild(enemySide)
}

// flowFor — яке поле читає юніт: те, що веде до ЧУЖОЇ сторони.
func (g *Game) flowFor(u *Pixel) *FlowField {
	if u.Faction == factionPlayer {
		return &g.flowToEnemySide
	}
	return &g.flowToPlayerSide
}

// drawFlowField — візуалізація (клавіша V): бачимо, як алгоритм «знає лабіринт».
// Стрілка в клітинці — куди веде поле; яскравість ∝ близькість до цілі.
//
// [КАМЕРА] Малюємо лише ВИДИМІ клітинки з камерою (px/py/s). Раніше кешувалось у
// позаекранний шар розміром screenWidth×screenHeight, але тепер світ може бути
// більшим за екран, і створювати worldWidth×worldHeight текстуру (100+ МБ) для
// дебаг-оверлея — марнотратство. Каллінг по камері компенсує: при зумі 1× видно
// стільки ж клітинок, скільки раніше; при зумі 4× — у 16 разів менше.
func (g *Game) drawFlowField(screen *ebiten.Image) {
	// Клавіша V циклює: 0 = вимкнено, 1 = поле ДО СТОРОНИ ГРАВЦЯ (його бачать
	// вороги), 2 = поле ДО ВОРОГІВ (його бачать твої юніти).
	f := &g.flowToPlayerSide
	if showFlowField == 2 {
		f = &g.flowToEnemySide
	}
	if !f.valid || f.maxDist == 0 {
		return
	}

	for r := 0; r < boidMapH; r++ {
		for c := 0; c < boidMapW; c++ {
			d := f.dist[r][c]
			if d == flowUnreachable {
				continue
			}
			// Світові координати центру клітинки
			wx := float32(c*pixelSize) + pixelSize/2
			wy := float32(r*pixelSize) + pixelSize/2

			// [КАМЕРА] Каллінг — не малюємо те, що за екраном.
			if !cam.visible(wx, wy) {
				continue
			}

			dx, dy := f.dirX[r][c], f.dirY[r][c]
			sx := cam.px(wx)
			sy := cam.py(wy)

			// Клітинка-джерело — без напрямку; позначаємо точкою.
			if dx == 0 && dy == 0 {
				hs := cam.s(2)
				vector.FillRect(screen, sx-hs, sy-hs, hs*2, hs*2, color.RGBA{255, 255, 255, 220}, false)
				continue
			}

			t := 1 - float32(d)/float32(f.maxDist) // 1 біля джерела → 0 на краю
			col := color.RGBA{
				R: uint8(80 + 175*t),
				G: uint8(170 + 85*t),
				B: 255,
				A: uint8(110 + 130*t),
			}

			arrowLen := cam.s(0.42 * pixelSize)
			tx := sx + dx*arrowLen
			ty := sy + dy*arrowLen
			vector.StrokeLine(screen, sx, sy, tx, ty, cam.s(1), col, false)
			ds := cam.s(1.5)
			vector.FillRect(screen, tx-ds, ty-ds, ds*2, ds*2, col, false)
		}
	}
}
