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

// FlowField — поле напрямків до гравця для всієї карти.
type FlowField struct {
	dist [boidMapH][boidMapW]int32   // кроків до гравця; flowUnreachable = недосяжно
	dirX [boidMapH][boidMapW]float32 // одиничний напрямок «куди йти»
	dirY [boidMapH][boidMapW]float32

	queue          []int32 // черга BFS, перевикористовується (без алокацій щокадру)
	srcCol, srcRow int     // клітинка гравця, від якої будували
	maxDist        int32   // найдальша досяжна клітинка (для розфарбування)
	valid          bool
}

// rebuild — будує поле від клітинки гравця. O(клітинок), ~2500 ітерацій.
func (f *FlowField) rebuild(srcCol, srcRow int) {
	f.valid = false
	f.maxDist = 0
	for r := 0; r < boidMapH; r++ {
		for c := 0; c < boidMapW; c++ {
			f.dist[r][c] = flowUnreachable
			f.dirX[r][c], f.dirY[r][c] = 0, 0
		}
	}
	if isWallAt(srcCol, srcRow) {
		return // гравець усередині стіни — поля нема (не має траплятись)
	}
	f.srcCol, f.srcRow = srcCol, srcRow
	f.dist[srcRow][srcCol] = 0

	// --- Крок 1: BFS-хвиля від гравця (відстань у клітинках) ---
	q := f.queue[:0]
	q = append(q, int32(srcRow*boidMapW+srcCol))
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
			// (клітинка гравця лишається (0,0) — ми вже на місці)
		}
	}
	f.valid = true
}

// dirAt — напрямок до гравця з точки (x,y) у пікселях. ok=false, якщо
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

// distAt — відстань у клітинках до гравця крізь лабіринт (−1 = недосяжно).
func (f *FlowField) distAt(x, y float32) int32 {
	c := int(x+pixelSize/2) / pixelSize
	r := int(y+pixelSize/2) / pixelSize
	if !f.valid || c < 0 || c >= boidMapW || r < 0 || r >= boidMapH {
		return flowUnreachable
	}
	return f.dist[r][c]
}

// updateFlowField — перебудовує поле, коли гравець перейшов в іншу клітинку.
// Стоїть на місці → поле лишається валідним, нічого не рахуємо.
func (g *Game) updateFlowField() {
	c := int(g.player.X+pixelSize/2) / pixelSize
	r := int(g.player.Y+pixelSize/2) / pixelSize
	if g.flow.valid && c == g.flow.srcCol && r == g.flow.srcRow {
		return
	}
	g.flow.rebuild(c, r)
}

// drawFlowField — візуалізація (клавіша V): бачимо, як алгоритм «знає лабіринт».
//
//	підсвітка клітинки — яскравість ∝ близькість до гравця (видно BFS-хвилю),
//	кожні flowBandStep кроків — світліша смуга (контурні «кільця» хвилі),
//	коротка стрілка — куди веде поле з цієї клітинки.
func (g *Game) drawFlowField(screen *ebiten.Image) {
	f := &g.flow
	if !f.valid || f.maxDist == 0 {
		return
	}
	// Заливку клітинок НЕ малюємо: контури рівної відстані в манхеттен-метриці —
	// це «ромби», у відкритому просторі вони читаються як шахове штрихування, а не
	// як кільця. Усю інформацію несуть стрілки: напрямок + яскравість = близькість.
	for r := 0; r < boidMapH; r++ {
		for c := 0; c < boidMapW; c++ {
			d := f.dist[r][c]
			if d == flowUnreachable {
				continue
			}
			dx, dy := f.dirX[r][c], f.dirY[r][c]
			cx := float32(c*pixelSize) + pixelSize/2
			cy := float32(r*pixelSize) + pixelSize/2

			// Клітинка гравця — без напрямку; позначаємо точкою.
			if dx == 0 && dy == 0 {
				vector.FillRect(screen, cx-2, cy-2, 4, 4, color.RGBA{255, 255, 255, 220}, false)
				continue
			}

			t := 1 - float32(d)/float32(f.maxDist) // 1 біля гравця → 0 на краю
			col := color.RGBA{
				R: uint8(80 + 175*t),
				G: uint8(170 + 85*t),
				B: 255,
				A: uint8(110 + 130*t),
			}

			// [ВАЖЛИВО] Хвіст рівно в центрі клітинки, голова — у напрямку руху,
			// плюс КРАПКА на кінці. Без крапки лінія симетрична, і напрямок
			// прочитати неможливо (той самий «/» = і NE, і SW) — тоді поле
			// виглядає як штрихування, а не як стрілки.
			const arrowLen = 0.42 * pixelSize
			tx := cx + dx*arrowLen
			ty := cy + dy*arrowLen
			vector.StrokeLine(screen, cx, cy, tx, ty, 1, col, false)
			vector.FillRect(screen, tx-1.5, ty-1.5, 3, 3, col, false)
		}
	}
}
