package main

import (
	"fmt"
	"os"
	"sort"
	"testing"
)

// [ПРОБА] Чому згладжування просідає САМЕ В БОЮ.
//
// Відтворюємо тут пакувальник атласа з ebiten (vector/atlas.go, setPaths), бо він
// детермінований і не потребує GPU. Три речі, які він робить і які пояснюють усе:
//
//	1. кожен ШЛЯХ отримує свою область атласа = його ГАБАРИТИ, округлені вгору до 16px;
//	2. при згладжуванні ширина області ПОДВОЮЄТЬСЯ (s.X *= 2);
//	3. області пакуються рядками в зображення 4093×4093; що не влізло — НОВЕ зображення.
//
// Плюс сам малюнок: при AA кожен шлях малюється у трафарет ВІСІМ разів (offsetAndColorsAA
// проти одного зсуву без AA).
//
// Звідси гіпотеза Євгена перевіряється прямо: якщо жало збільшує ГАБАРИТИ шляху
// щупальця, площа атласа росте — і росте нелінійно, бо діагональ дає прямокутник.
//
// Запуск: BOIDS_AA=1 go test -run TestAAAtlasCost -v

func roundUp16probe(x int) int { return (x + 15) / 16 * 16 }

type pathBox struct {
	name string
	w, h int
}

// packAtlas — точна копія алгоритму з ebiten: сортування за висотою (спадання), потім
// за шириною (зростання), і укладання рядками.
func packAtlas(boxes []pathBox, antialias bool) (images int, totalArea int) {
	type reg struct{ w, h int }
	regs := make([]reg, len(boxes))
	for i, b := range boxes {
		w := roundUp16probe(b.w)
		h := roundUp16probe(b.h)
		if antialias {
			w *= 2 // ← ось воно: при згладжуванні область удвічі ширша
		}
		regs[i] = reg{w, h}
	}
	sort.SliceStable(regs, func(i, j int) bool {
		if regs[i].h != regs[j].h {
			return regs[i].h > regs[j].h
		}
		return regs[i].w < regs[j].w
	})

	const maxSize = 4093
	sizes := []struct{ x, y int }{{}}
	idx, rowH, posX, posY := 0, 0, 0, 0
	for i, r := range regs {
		if i == 0 {
			rowH = r.h
		} else if posX+r.w > maxSize {
			posX = 0
			posY += rowH
			if posY+r.h > maxSize {
				idx++
				sizes = append(sizes, struct{ x, y int }{})
				posY = 0
				rowH = r.h
			} else if r.h > rowH {
				rowH = r.h
			}
		}
		if posX+r.w > sizes[idx].x {
			sizes[idx].x = posX + r.w
		}
		if posY+r.h > sizes[idx].y {
			sizes[idx].y = posY + r.h
		}
		posX += r.w
	}
	for _, s := range sizes {
		totalArea += s.x * s.y
	}
	return len(sizes), totalArea
}

// unitBoxes — габарити всіх шляхів одного юніта, як їх бачить атлас.
//
// Шляхи рахуємо ті самі, що малює рендер: furWidthGroups груп ворсу, один шлях кінцівок,
// один шлях щупальця, одне тіло.
func unitBoxes(p *Pixel) []pathBox {
	cx, cy := p.X+pixelSize/2, p.Y+pixelSize/2
	scale := bodyScaleOf(p)

	box := func(name string, pts [][2]float32, pad float32) pathBox {
		if len(pts) == 0 {
			return pathBox{name, 0, 0}
		}
		minX, minY := pts[0][0], pts[0][1]
		maxX, maxY := minX, minY
		for _, q := range pts {
			if q[0] < minX {
				minX = q[0]
			}
			if q[0] > maxX {
				maxX = q[0]
			}
			if q[1] < minY {
				minY = q[1]
			}
			if q[1] > maxY {
				maxY = q[1]
			}
		}
		return pathBox{name,
			int(cam.s(maxX-minX+pad)) + 1,
			int(cam.s(maxY-minY+pad)) + 1}
	}

	var out []pathBox

	// Ворс: групи за товщиною, як у drawFur.
	fx, fy := furRoot(cx, cy, scale)
	for g := 0; g < furWidthGroups; g++ {
		var pts [][2]float32
		for i := 0; i < furStrands; i++ {
			px, py := fx, fy
			for j := 0; j < furJoints; j++ {
				jx, jy := p.Fur[i][j][0], p.Fur[i][j][1]
				if j*furWidthGroups/furJoints == g {
					pts = append(pts, [2]float32{px, py}, [2]float32{jx, jy})
				}
				px, py = jx, jy
			}
		}
		out = append(out, box(fmt.Sprintf("ворс%d", g), pts, furWidthRoot))
	}

	// Кінцівки — один шлях на всі чотири.
	var lp [][2]float32
	for i := 0; i < limbCount; i++ {
		rx, ry := limbRoot(i, cx, cy, scale)
		lp = append(lp, [2]float32{rx, ry}, [2]float32{p.Limbs[i][0], p.Limbs[i][1]})
	}
	out = append(out, box("кінцівки", lp, limbWidth))

	// Щупальце — ОСЬ ГОЛОВНИЙ ПІДОЗРЮВАНИЙ.
	tx, ty := tentRoot(cx, cy, scale)
	tp := [][2]float32{{tx, ty}}
	for i, n := 0, tentJointsOf(p); i < n; i++ {
		tp = append(tp, [2]float32{p.Tent[i][0], p.Tent[i][1]})
	}
	out = append(out, box("щупальце", tp, tentWidth))

	// Тіло.
	vx, vy := bodyVertices(*p, cx, cy, scale)
	var bp [][2]float32
	for i := 0; i < brainWhiskers; i++ {
		bp = append(bp, [2]float32{vx[i], vy[i]})
	}
	out = append(out, box("тіло", bp, 0))

	return out
}

// settle — прокрутити анімацію, щоб відросток зайняв усталене положення.
func settle(p *Pixel, frames int) {
	for i := 0; i < frames; i++ {
		updateFur(p)
		updateBalls(p)
		updateTentacle(p)
		updateLimbs(p)
		updateBody(p)
	}
}

func TestAAAtlasCost(t *testing.T) {
	if os.Getenv("BOIDS_AA") == "" {
		t.Skip("проба згладжування: BOIDS_AA=1")
	}
	cam.zoom = camZoomMin
	cam.cx, cam.cy = worldWidth/2, worldHeight/2

	mk := func(sting bool) *Pixel {
		p := &Pixel{X: cam.cx, Y: cam.cy, Cfg: ConfigWarden, HP: 3, MaxHP: 3}
		p.resetFur()
		settle(p, 200)
		if sting {
			// Жало під 45°: найгірший випадок для габаритів, бо діагональ дає
			// прямокутник, а не смужку.
			p.StingDirX, p.StingDirY = 0.707, -0.707
			p.StingTimer = stingFrames + stingCooldown
			settle(p, stingFrames/2) // середина активної фази
		}
		return p
	}

	rest, sting := mk(false), mk(true)

	fmt.Printf("\n=== габарити шляхів ОДНОГО юніта (екранні px) ===\n")
	fmt.Printf("%-12s %-16s %-16s\n", "шлях", "спокій", "жало")
	rb, sb := unitBoxes(rest), unitBoxes(sting)
	for i := range rb {
		fmt.Printf("%-12s %5d×%-10d %5d×%-10d\n", rb[i].name, rb[i].w, rb[i].h, sb[i].w, sb[i].h)
	}

	tentR, tentS := rb[len(rb)-2], sb[len(sb)-2]
	aR := roundUp16probe(tentR.w) * roundUp16probe(tentR.h)
	aS := roundUp16probe(tentS.w) * roundUp16probe(tentS.h)
	fmt.Printf("\nплоща області ЩУПАЛЬЦЯ в атласі: спокій %d px², жало %d px²  ×%.1f\n",
		aR, aS, float64(aS)/float64(aR))

	// Сцена: N юнітів, частина з них у жалі.
	scene := func(n, stingingN int) []pathBox {
		var all []pathBox
		for i := 0; i < n; i++ {
			src := rest
			if i < stingingN {
				src = sting
			}
			cp := *src
			cp.X = cam.cx + float32((i%10)*90) - 450
			cp.Y = cam.cy + float32((i/10)*70) - 175
			for j := range cp.Fur {
				for k := range cp.Fur[j] {
					cp.Fur[j][k][0] += cp.X - src.X
					cp.Fur[j][k][1] += cp.Y - src.Y
				}
			}
			for j := range cp.Tent {
				cp.Tent[j][0] += cp.X - src.X
				cp.Tent[j][1] += cp.Y - src.Y
			}
			for j := range cp.Limbs {
				cp.Limbs[j][0] += cp.X - src.X
				cp.Limbs[j][1] += cp.Y - src.Y
			}
			all = append(all, unitBoxes(&cp)...)
		}
		return all
	}

	fmt.Printf("\n=== атлас на кадр: 50 юнітів ===\n")
	fmt.Printf("%-22s %-8s %-14s %-14s %s\n", "сцена", "шляхів", "зображень", "площа атласа", "малювань у трафарет")
	for _, c := range []struct {
		name      string
		stingingN int
	}{{"спокій (0 жал)", 0}, {"сутичка (10 жал)", 10}, {"бійня (30 жал)", 30}, {"усі 50 у жалі", 50}} {
		boxes := scene(50, c.stingingN)
		for _, aa := range []bool{false, true} {
			imgs, area := packAtlas(boxes, aa)
			mark := "без AA"
			calls := len(boxes)
			if aa {
				mark = "З AA  "
				calls = len(boxes) * 8
			}
			fmt.Printf("%-22s %-8d %s %-6d %-14s %d\n",
				c.name+" "+mark, len(boxes), " ", imgs,
				fmt.Sprintf("%.1f Мпx", float64(area)/1e6), calls)
		}
	}

	// Окремо: чи не шкодить наш pinAARegion? Він додає шлях НА ВЕСЬ ЕКРАН, тобто
	// область 1712×992, а при AA ще й удвічі ширшу — 3424 з 4093 доступних.
	fmt.Printf("\n=== вплив pinAARegion (шлях на весь екран) ===\n")
	base := scene(50, 30)
	withPin := append(append([]pathBox{}, base...), pathBox{"рамка", screenWidth, screenHeight})
	for _, tc := range []struct {
		name  string
		boxes []pathBox
	}{{"без рамки", base}, {"з рамкою", withPin}} {
		imgs, area := packAtlas(tc.boxes, true)
		fmt.Printf("%-12s зображень %d, площа %.1f Мпx\n", tc.name, imgs, float64(area)/1e6)
	}
}
