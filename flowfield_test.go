package main

import (
	"math"
	"testing"
)

// TestFlowFieldInvariant перевіряє КОРЕКТНІСТЬ поля потоку строго, а не на око.
//
// Ключовий інваріант: у кожній досяжній клітинці стрілка мусить вести на сусіда
// зі СТРОГО меншою відстанню. Тоді, йдучи за стрілками, ми гарантовано дійдемо
// до гравця (dist монотонно спадає до 0) і ніде не зациклимось.
//
// [ЧОМУ НЕ «рівно на 1»] BFS рахує відстань по 4 сусідах (манхеттен), а напрямок
// обираємо по 8 — тож ДІАГОНАЛЬНИЙ крок скорочує манхеттен-відстань на 2. Це
// не помилка, а бажаний «зріз кута»: реальний шлях виходить коротшим.
//
// Запуск: go test -run TestFlowField -v
func TestFlowFieldInvariant(t *testing.T) {
	// Гравець — у першій відкритій клітинці рівня (не залежить від дизайну карти).
	pc, pr := -1, -1
	for r := 0; r < boidMapH && pr < 0; r++ {
		for c := 0; c < boidMapW; c++ {
			if !isWallAt(c, r) {
				pc, pr = c, r
				break
			}
		}
	}
	if pr < 0 {
		t.Fatal("на рівні немає жодної відкритої клітинки")
	}

	var f FlowField
	f.rebuildFrom(pc, pr)
	if !f.valid {
		t.Fatal("поле не побудувалось")
	}
	if f.dist[pr][pc] != 0 {
		t.Fatalf("клітинка гравця має dist=0, а не %d", f.dist[pr][pc])
	}

	reachable := 0
	for r := 0; r < boidMapH; r++ {
		for c := 0; c < boidMapW; c++ {
			d := f.dist[r][c]
			if d == flowUnreachable {
				continue
			}
			reachable++
			if isWallAt(c, r) {
				t.Fatalf("стіна (%d,%d) отримала відстань %d", c, r, d)
			}
			dx, dy := f.dirX[r][c], f.dirY[r][c]
			if dx == 0 && dy == 0 {
				// Лише клітинка гравця може бути без напрямку.
				if d != 0 {
					t.Fatalf("клітинка (%d,%d) з dist=%d не має напрямку", c, r, d)
				}
				continue
			}
			// Стрілка вирівняна по сітці → округлення дає цілий зсув сусіда.
			nc := c + int(math.Round(float64(dx)))
			nr := r + int(math.Round(float64(dy)))
			if isWallAt(nc, nr) {
				t.Fatalf("стрілка з (%d,%d) веде у стіну (%d,%d)", c, r, nc, nr)
			}
			nd := f.dist[nr][nc]
			if nd == flowUnreachable || nd >= d {
				t.Fatalf("стрілка з (%d,%d) dist=%d веде в (%d,%d) dist=%d — відстань не спадає",
					c, r, d, nc, nr, nd)
			}
			// Діагональ може скоротити на 2 (манхеттен), ортогональ — рівно на 1.
			diag := int(math.Round(float64(dx))) != 0 && int(math.Round(float64(dy))) != 0
			if !diag && nd != d-1 {
				t.Fatalf("ортогональна стрілка з (%d,%d) dist=%d → dist=%d (мало бути %d)",
					c, r, d, nd, d-1)
			}
		}
	}
	t.Logf("поле коректне: %d досяжних клітинок, максимальна відстань %d кроків",
		reachable, f.maxDist)
}

// TestFlowFieldWalksToPlayer — практична перевірка: з КОЖНОЇ досяжної клітинки
// йдемо за стрілками й мусимо дійти до гравця за скінченну кількість кроків.
func TestFlowFieldWalksToPlayer(t *testing.T) {
	pc, pr := boidMapW/2, boidMapH/2
	if isWallAt(pc, pr) { // центр у стіні — беремо будь-яку відкриту
		for r := 0; r < boidMapH; r++ {
			for c := 0; c < boidMapW; c++ {
				if !isWallAt(c, r) {
					pc, pr = c, r
					r, c = boidMapH, boidMapW
				}
			}
		}
	}
	var f FlowField
	f.rebuildFrom(pc, pr)

	worst := int32(0)
	for r := 0; r < boidMapH; r++ {
		for c := 0; c < boidMapW; c++ {
			if f.dist[r][c] == flowUnreachable {
				continue
			}
			cc, rr := c, r
			steps := int32(0)
			for f.dist[rr][cc] != 0 {
				dx, dy := f.dirX[rr][cc], f.dirY[rr][cc]
				cc += int(math.Round(float64(dx)))
				rr += int(math.Round(float64(dy)))
				steps++
				if steps > f.maxDist+1 {
					t.Fatalf("з (%d,%d) не дійшли до гравця за %d кроків (зациклились?)", c, r, steps)
				}
			}
			if cc != pc || rr != pr {
				t.Fatalf("з (%d,%d) прийшли в (%d,%d), а гравець у (%d,%d)", c, r, cc, rr, pc, pr)
			}
			if steps > worst {
				worst = steps
			}
		}
	}
	t.Logf("з усіх клітинок шлях знайдено; найдовший — %d кроків", worst)
}

// TestKillerInputsUseFlowNotDirect — [ВБИВЦЯ] суть кроку 2b-2: коли між агентом
// і ціллю СТІНА, напрямок на вході має вести в ОБХІД (flow-field), а не прямо
// в стіну. Саме через це вбивця ходить коридорами, а реактивний рій — тикається.
func TestKillerInputsUseFlowNotDirect(t *testing.T) {
	// Штучна карта: суцільна вертикальна стіна з прорізом угорі.
	saved := tileMap
	defer func() { tileMap = saved }()
	tileMap = [boidMapH][boidMapW]bool{}
	const wallCol = 20
	for r := 2; r < boidMapH; r++ { // проріз — у рядках 0..1
		tileMap[r][wallCol] = true
	}

	// Ціль ліворуч від стіни, вбивця праворуч, обидва внизу.
	toPx := func(c, r int) (float32, float32) {
		return float32(c * pixelSize), float32(r * pixelSize)
	}
	px, py := toPx(wallCol-3, boidMapH-3)
	ex, ey := toPx(wallCol+3, boidMapH-3)

	var flow FlowField
	flow.rebuildFrom((int(px)+pixelSize/2)/pixelSize, (int(py)+pixelSize/2)/pixelSize)
	if !flow.valid {
		t.Fatal("поле не побудувалось")
	}

	killer := &Pixel{X: ex, Y: ey, Cfg: ConfigKiller}
	player := &Pixel{X: px, Y: py}
	in := GatherKillerInputs(killer, player, &flow)

	dirX, dirY := in[inDirX], in[inDirX+1]
	if dirX == 0 && dirY == 0 {
		t.Fatal("вбивця не отримав напрямку від flow-field")
	}
	// Прямий напрямок — ліворуч (у стіну). Правильний обхід — УГОРУ до прорізу.
	if dirX < 0 && dirY >= 0 {
		t.Fatalf("напрямок веде в стіну (прямо на ціль): dir=(%.2f, %.2f)", dirX, dirY)
	}
	if dirY >= 0 {
		t.Fatalf("обхід мав вести вгору до прорізу, а веде dir=(%.2f, %.2f)", dirX, dirY)
	}

	// Відстань — ПО ЛАБІРИНТУ: обхід довгий, тож помітно більший за прямий шлях.
	directCells := float32(6) // 6 клітинок по прямій
	if got := in[inDist] * flowDistNorm; got <= directCells*2 {
		t.Fatalf("відстань %.1f клітинок схожа на пряму (%.0f), а мала бути по лабіринту",
			got, directCells)
	}
	t.Logf("обхід: dir=(%.2f, %.2f), відстань по лабіринту ≈ %.0f клітинок (прямо було б %.0f)",
		dirX, dirY, in[inDist]*flowDistNorm, directCells)
}

// TestKillerRewardFollowsCorridor — [ВБИВЦЯ] крок 2b-4: коли ціль за стіною,
// правильний рух (в обхід, УЗДОВЖ коридору) має давати ДОДАТНУ нагороду, хоча
// пряма відстань при цьому РОСТЕ. Без цього вхід і нагорода суперечать: поле
// каже «йди в обхід», а нагорода штрафує саме за це — і агент тикається в стіну.
func TestKillerRewardFollowsCorridor(t *testing.T) {
	saved := tileMap
	defer func() { tileMap = saved }()
	tileMap = [boidMapH][boidMapW]bool{}
	const wallCol = 20
	for r := 2; r < boidMapH; r++ { // суцільна стіна, проріз угорі
		tileMap[r][wallCol] = true
	}

	toPx := func(c, r int) (float32, float32) {
		return float32(c * pixelSize), float32(r * pixelSize)
	}
	px, py := toPx(wallCol-3, boidMapH-3) // ціль ліворуч від стіни
	ex, ey := toPx(wallCol+3, boidMapH-3) // вбивця праворуч

	var flow FlowField
	flow.rebuildFrom((int(px)+pixelSize/2)/pixelSize, (int(py)+pixelSize/2)/pixelSize)

	// Вбивця рухається ВГОРУ — до прорізу. Пряма відстань до цілі при цьому росте.
	killer := &Pixel{X: ex, Y: ey, VelY: -ConfigKiller.MaxSpeed, Cfg: ConfigKiller,
		HP: 3, MaxHP: 3, Brain: NewBrain()}
	killer.Brain.flowNav = true
	player := &Pixel{X: px, Y: py}

	GatherKillerInputs(killer, player, &flow) // побічно рахує flowProgress

	if killer.Brain.progress <= 0 {
		t.Fatalf("рух до прорізу мав дати ДОДАТНИЙ прогрес, а дав %.3f", killer.Brain.progress)
	}

	// А тепер той самий стан очима СТАРОЇ (прямої) метрики: вона б сказала «гірше».
	dx, dy := px-ex, py-ey
	distNow := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	afterY := ey + killer.VelY
	dyAfter := py - afterY
	distAfter := float32(math.Sqrt(float64(dx*dx + dyAfter*dyAfter)))
	if distAfter <= distNow {
		t.Fatalf("для чистоти тесту пряма відстань мала ЗРОСТИ: %.1f → %.1f", distNow, distAfter)
	}

	// Нагорода за flow-метрикою — додатна, попри зростання прямої відстані.
	killer.Brain.hasPrev = true
	r := killer.Brain.rewardFor(false, 0)
	if r <= 0 {
		t.Fatalf("нагорода за правильний обхід має бути > 0, а вона %.3f", r)
	}
	t.Logf("обхід: прогрес коридором +%.2f → нагорода %+.3f (пряма відстань при цьому %.2f→%.2f — ЗРОСЛА)",
		killer.Brain.progress, r, distNow, distAfter)
}

// TestFlowFieldMultiSource — [КОМАНДИ] перевіряє серце симетричних боїв: поле з
// БАГАТЬМА джерелами. Кожна клітинка має вести до НАЙБЛИЖЧОГО джерела лабіринтом,
// а не до якогось одного. Без цього «свій вбивця» був би фікцією — він бігав би
// до чужої цілі.
func TestFlowFieldMultiSource(t *testing.T) {
	saved := tileMap
	defer func() { tileMap = saved }()
	tileMap = [boidMapH][boidMapW]bool{} // відкрите поле, без стін

	left := flowSource{5, boidMapH / 2}
	right := flowSource{boidMapW - 6, boidMapH / 2}

	var f FlowField
	f.rebuild([]flowSource{left, right})
	if !f.valid {
		t.Fatal("поле не побудувалось")
	}

	// Обидва джерела мають dist=0 — хвиля стартувала з кожного.
	if f.dist[left.row][left.col] != 0 || f.dist[right.row][right.col] != 0 {
		t.Fatalf("не всі джерела на нулі: left=%d right=%d",
			f.dist[left.row][left.col], f.dist[right.row][right.col])
	}

	// Клітинка біля ЛІВОГО джерела має вести ВЛІВО, біля правого — ВПРАВО.
	nearLeftX := float32((left.col + 3) * pixelSize)
	nearLeftY := float32(left.row * pixelSize)
	if dx, _, ok := f.dirAt(nearLeftX, nearLeftY); !ok || dx >= 0 {
		t.Fatalf("біля лівого джерела напрямок мав бути вліво, а dx=%.2f (ok=%v)", dx, ok)
	}
	nearRightX := float32((right.col - 3) * pixelSize)
	nearRightY := float32(right.row * pixelSize)
	if dx, _, ok := f.dirAt(nearRightX, nearRightY); !ok || dx <= 0 {
		t.Fatalf("біля правого джерела напрямок мав бути вправо, а dx=%.2f (ok=%v)", dx, ok)
	}

	// Відстань посередині ≈ половина шляху між джерелами (хвилі зустрілись).
	midX := float32((left.col + right.col) / 2 * pixelSize)
	midY := float32(left.row * pixelSize)
	mid := f.distAt(midX, midY)
	halfGap := int32((right.col - left.col) / 2)
	if mid < halfGap-2 || mid > halfGap+2 {
		t.Fatalf("посередині відстань %d, очікували ≈%d (хвилі мали зустрітись)", mid, halfGap)
	}
	t.Logf("дві хвилі зустрілись посередині: dist=%d (≈%d), maxDist=%d", mid, halfGap, f.maxDist)
}
