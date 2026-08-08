package main

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// [ПРОБА] Тимчасовий профіль МАЛЮВАННЯ без GPU.
//
// Малювати headless не можна, але найдорожча частина вектора робиться на CPU:
// шлях треба ТЕСЕЛЮВАТИ в трикутники (AppendVerticesAndIndicesForStroke). Саме це
// й міряємо — окремо на ворс, щупальце, кінцівки, тіло. Плюс рахуємо видимі тайли
// стін, бо їх кількість залежить від СВІТУ, а не від юнітів.
//
// Запуск: BOIDS_RENDER=1 go test -run TestRenderCost -v

func tessCost(build func(*vector.Path), stroke bool, width float32, reps int) (ns int64, verts int) {
	var vs []ebiten.Vertex
	var is []uint16
	var p vector.Path
	build(&p)
	// один прогін поза заміром — прогріти
	if stroke {
		vs, is = p.AppendVerticesAndIndicesForStroke(vs[:0], is[:0], &vector.StrokeOptions{Width: width})
	} else {
		vs, is = p.AppendVerticesAndIndicesForFilling(vs[:0], is[:0])
	}
	_ = is
	verts = len(vs)

	s := time.Now()
	for r := 0; r < reps; r++ {
		var q vector.Path
		build(&q)
		if stroke {
			vs, is = q.AppendVerticesAndIndicesForStroke(vs[:0], is[:0], &vector.StrokeOptions{Width: width})
		} else {
			vs, is = q.AppendVerticesAndIndicesForFilling(vs[:0], is[:0])
		}
	}
	return time.Since(s).Nanoseconds() / int64(reps), verts
}

func TestRenderCost(t *testing.T) {
	if os.Getenv("BOIDS_RENDER") == "" {
		t.Skip("профіль малювання: BOIDS_RENDER=1")
	}
	const reps = 3000

	var p Pixel
	p.X, p.Y = worldWidth/2, worldHeight/2
	p.Cfg = ConfigWarden
	p.HP, p.MaxHP = 3, 3
	p.resetFur()
	cam.zoom = camZoomMin
	cam.cx, cam.cy = p.X, p.Y

	cx, cy := p.X+pixelSize/2, p.Y+pixelSize/2
	scale := bodyScaleOf(&p)
	fx, fy := furRoot(cx, cy, scale)

	// Ворс — рівно як у drawFur: furWidthGroups шляхів, у сумі furStrands×furJoints
	// сегментів. Міряємо ВСІ групи разом, бо на кадр вони всі й будуються.
	furNs := int64(0)
	furVerts := 0
	for gi := 0; gi < furWidthGroups; gi++ {
		g := gi
		ns, v := tessCost(func(path *vector.Path) {
			for i := 0; i < furStrands; i++ {
				px, py := fx, fy
				for j := 0; j < furJoints; j++ {
					if j*furWidthGroups/furJoints != g {
						px, py = p.Fur[i][j][0], p.Fur[i][j][1]
						continue
					}
					jx, jy := p.Fur[i][j][0], p.Fur[i][j][1]
					path.MoveTo(cam.px(px), cam.py(py))
					path.LineTo(cam.px(jx), cam.py(jy))
					px, py = jx, jy
				}
			}
		}, true, cam.s(furWidthRoot), reps)
		furNs += ns
		furVerts += v
	}

	n := tentJointsOf(&p)
	tentNs, tentVerts := tessCost(func(path *vector.Path) {
		px, py := tentRoot(cx, cy, scale)
		path.MoveTo(cam.px(px), cam.py(py))
		for j := 0; j < n; j++ {
			path.LineTo(cam.px(p.Tent[j][0]), cam.py(p.Tent[j][1]))
		}
	}, true, cam.s(tentWidth), reps)

	limbNs, limbVerts := tessCost(func(path *vector.Path) {
		for i := 0; i < limbCount; i++ {
			rx, ry := limbRoot(i, cx, cy, scale)
			path.MoveTo(cam.px(rx), cam.py(ry))
			path.LineTo(cam.px(p.Limbs[i][0]), cam.py(p.Limbs[i][1]))
		}
	}, true, cam.s(limbWidth), reps)

	bodyNs, bodyVerts := tessCost(func(path *vector.Path) {
		vx, vy := bodyVertices(p, cx, cy, scale)
		path.MoveTo(cam.px(vx[0]), cam.py(vy[0]))
		for i := 1; i < brainWhiskers; i++ {
			path.LineTo(cam.px(vx[i]), cam.py(vy[i]))
		}
		path.Close()
	}, false, 0, reps)

	type row struct {
		name  string
		ns    int64
		verts int
	}
	rows := []row{
		{"ворс", furNs, furVerts},
		{"щупальце", tentNs, tentVerts},
		{"кінцівки", limbNs, limbVerts},
		{"тіло", bodyNs, bodyVerts},
	}
	var sum int64
	for _, r := range rows {
		sum += r.ns
	}

	fmt.Printf("\n=== тесселяція ОДНОГО юніта (CPU, без GPU) ===\n")
	for _, r := range rows {
		fmt.Printf("  %-10s %8.1f мкс  %5.1f%%  вершин: %d\n",
			r.name, float64(r.ns)/1000, float64(r.ns)/float64(sum)*100, r.verts)
	}
	fmt.Printf("  %-10s %8.1f мкс\n", "разом", float64(sum)/1000)

	budget := 1000.0 / 120.0
	for _, units := range []int{25, 50} {
		ms := float64(sum) * float64(units) / 1e6
		fmt.Printf("\n%d юнітів: %.2f мс тесселяції на кадр = %.0f%% бюджету 120 FPS (%.2f мс)\n",
			units, ms, ms/budget*100, budget)
	}

	// Стіни: скільки тайлів у кадрі при зумі 1.0 (кількість залежить від СВІТУ).
	total, vis := 0, 0
	for row := 0; row < boidMapH; row++ {
		for col := 0; col < boidMapW; col++ {
			if tileMap[row][col] {
				total++
				if cam.visible(float32(col*pixelSize), float32(row*pixelSize)) {
					vis++
				}
			}
		}
	}
	fmt.Printf("\nстіни: %d тайлів у світі, %d у кадрі при зумі 1.0\n", total, vis)
	fmt.Printf("  було: %d викликів vector (FillRect+StrokeRect на тайл)\n", vis*2)
	fmt.Printf("  стало: 2 виклики (FillPath+StrokePath на спільному шляху)\n")
	fmt.Printf("виклики на юніта: ворс %d + кінцівки %d + щупальце %d + кульки %d + тіло 1 + HP 2 = %d\n",
		furWidthGroups, 1+limbCount, 2, ballCount, furWidthGroups+1+limbCount+2+ballCount+3)
}
