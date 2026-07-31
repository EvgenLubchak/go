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
	f.rebuild(pc, pr)
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
	f.rebuild(pc, pr)

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
