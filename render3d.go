package main

import (
	"image/color"
	"math"
	"sort"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// ==========================================================================
// RAYCASTER — псевдо-3D вид від першої особи (техніка Wolfenstein 3D, 1992).
//
// Ідея: світ лишається 2D-сіткою стін (tileMap). З позиції гравця під кутом
// camAngle кидаємо по одному променю НА КОЖНУ КОЛОНКУ екрана в межах поля зору
// (FOV). Промінь марширує по сітці (DDA), поки не влучить у стіну; відстань до
// стіни задає ВИСОТУ вертикальної смуги (ближче → вища) → ілюзія глибини.
//
// Це та сама математика, що й wallWhisker (промінь по tileMap до стіни), тільки
// променів багато й малюємо смуги замість «близькості».
// ==========================================================================

const (
	rayFOV       = math.Pi / 3 // поле зору ~60°
	rayColW      = 4           // ширина смуги в пікселях (менше = чіткіше, дорожче)
	rayMaxCell   = 80          // стеля кроків DDA (щоб не зациклитись)
	rayViewCells = 26.0        // дальність для затемнення (клітинки)
	spriteScale  = 0.8         // розмір ворога-спрайта відносно стіни-клітинки
)

// castColumn — DDA-марш променя по сітці стін. Повертає ПЕРПЕНДИКУЛЯРНУ відстань
// до стіни (в клітинках) і бік стіни (0 = вертикальна грань, 1 = горизонтальна).
// posX,posY — позиція гравця в КЛІТИНКАХ; rayDirX,rayDirY — напрямок променя.
//
// [GO: DDA] Крокуємо не по пікселях, а від грані до грані сітки — тому один крок
// = одна клітинка, і промінь долітає до стіни за ~десяток кроків (швидко).
func castColumn(posX, posY, rayDirX, rayDirY float32) (perpDist float32, side int, boundary bool) {
	mapX, mapY := int(posX), int(posY)

	deltaX := float32(1e30)
	if rayDirX != 0 {
		deltaX = float32(math.Abs(1 / float64(rayDirX)))
	}
	deltaY := float32(1e30)
	if rayDirY != 0 {
		deltaY = float32(math.Abs(1 / float64(rayDirY)))
	}

	var stepX, stepY int
	var sideDistX, sideDistY float32
	if rayDirX < 0 {
		stepX = -1
		sideDistX = (posX - float32(mapX)) * deltaX
	} else {
		stepX = 1
		sideDistX = (float32(mapX) + 1 - posX) * deltaX
	}
	if rayDirY < 0 {
		stepY = -1
		sideDistY = (posY - float32(mapY)) * deltaY
	} else {
		stepY = 1
		sideDistY = (float32(mapY) + 1 - posY) * deltaY
	}

	for i := 0; i < rayMaxCell; i++ {
		if sideDistX < sideDistY {
			sideDistX += deltaX
			mapX += stepX
			side = 0
		} else {
			sideDistY += deltaY
			mapY += stepY
			side = 1
		}
		if isWallAt(mapX, mapY) {
			// Межа рівня = вихід за сітку (крізь неї гравець робить wrap).
			// Справжня стіна = тайл усередині сітки.
			boundary = mapX < 0 || mapX >= boidMapW || mapY < 0 || mapY >= boidMapH
			break
		}
	}

	// (sideDist - delta) — перпендикулярна відстань до площини камери (без fisheye,
	// бо промені задані через площину камери — див. planeX/planeY нижче).
	if side == 0 {
		perpDist = sideDistX - deltaX
	} else {
		perpDist = sideDistY - deltaY
	}
	if perpDist < 0.001 {
		perpDist = 0.001
	}
	return
}

// drawFirstPerson малює сцену від першої особи (крок 1: лише стіни; вороги-
// спрайти — крок 2). Симуляція не чіпається — це суто інший РЕНДЕР тих самих даних.
func (g *Game) drawFirstPerson(screen *ebiten.Image) {
	// Небо (верх) і підлога (низ).
	vector.FillRect(screen, 0, 0, screenWidth, screenHeight/2, color.RGBA{22, 22, 38, 255}, false)
	vector.FillRect(screen, 0, screenHeight/2, screenWidth, screenHeight/2, color.RGBA{32, 32, 38, 255}, false)

	posX := (g.player.X + pixelSize/2) / pixelSize
	posY := (g.player.Y + pixelSize/2) / pixelSize
	dirX := float32(math.Cos(float64(g.camAngle)))
	dirY := float32(math.Sin(float64(g.camAngle)))
	// Площина камери: перпендикуляр до напрямку × tan(FOV/2). Задання променів
	// через (dir + plane·cameraX) автоматично дає перпендикулярну відстань → без fisheye.
	t := float32(math.Tan(rayFOV / 2))
	planeX, planeY := -dirY*t, dirX*t

	// z-буфер: перпендикулярна відстань до стіни в КОЖНІЙ смузі. Потрібен, щоб
	// спрайти-вороги коректно ховалися за стінами (тест глибини по колонках).
	var zbuf [screenWidth / rayColW]float32

	cols := screenWidth / rayColW
	for c := 0; c < cols; c++ {
		cameraX := 2*float32(c)/float32(cols) - 1 // -1..+1 поперек екрана
		rayDirX := dirX + planeX*cameraX
		rayDirY := dirY + planeY*cameraX

		perpDist, side, boundary := castColumn(posX, posY, rayDirX, rayDirY)
		zbuf[c] = perpDist

		lineH := float32(screenHeight) / perpDist
		y0 := (float32(screenHeight) - lineH) / 2

		// Затемнення: далі → темніше; горизонтальна грань (side==1) ще темніша.
		shade := 1 - perpDist/rayViewCells
		if shade < 0.15 {
			shade = 0.15
		}
		if side == 1 {
			shade *= 0.7
		}
		var col color.RGBA
		if boundary {
			// Межа рівня (крізь неї працює wrap) — тепла бурштинова позначка.
			col = color.RGBA{uint8(160*shade) + 50, uint8(95*shade) + 25, uint8(25*shade) + 10, 255}
		} else {
			// Звичайна стіна — сіро-блакитна.
			col = color.RGBA{uint8(110*shade) + 30, uint8(110*shade) + 30, uint8(150*shade) + 40, 255}
		}
		vector.FillRect(screen, float32(c*rayColW), y0, rayColW, lineH, col, false)
	}

	g.drawSprites(screen, zbuf[:], posX, posY, dirX, dirY, planeX, planeY)
}

// drawSprites малює ворогів як БІЛБОРДИ (пласкі спрайти, завжди «обличчям» до
// камери) з тестом глибини проти стін (z-буфер по смугах).
//
// [RAYCASTER: SPRITE CASTING]
// Кожного ворога переводимо в систему координат камери (інверсна матриця
// [dir|plane]). transformY — глибина (перпендикуляр); transformX — зсув убік.
// Далі: екранна X-позиція, розмір ∝ 1/глибина, і малюємо вертикальними смугами,
// пропускаючи ті, що ЗА стіною (tY ≥ zbuf[смуга]).
func (g *Game) drawSprites(screen *ebiten.Image, zbuf []float32, posX, posY, dirX, dirY, planeX, planeY float32) {
	if len(g.units) == 0 {
		return
	}

	// Порядок далеко→близько (painter's): дальні першими, щоб ближчі перекривали.
	order := make([]int, len(g.units))
	d2 := make([]float32, len(g.units))
	for i := range g.units {
		order[i] = i
		rx := (g.units[i].X+pixelSize/2)/pixelSize - posX
		ry := (g.units[i].Y+pixelSize/2)/pixelSize - posY
		d2[i] = rx*rx + ry*ry
	}
	sort.Slice(order, func(a, b int) bool { return d2[order[a]] > d2[order[b]] })

	invDet := 1 / (planeX*dirY - dirX*planeY)
	for _, i := range order {
		e := &g.units[i]
		relX := (e.X+pixelSize/2)/pixelSize - posX
		relY := (e.Y+pixelSize/2)/pixelSize - posY

		tX := invDet * (dirY*relX - dirX*relY)
		tY := invDet * (-planeY*relX + planeX*relY) // глибина
		if tY <= 0.1 {
			continue // позаду камери або впритул
		}

		screenX := (float32(screenWidth) / 2) * (1 + tX/tY)
		size := (float32(screenHeight) / tY) * spriteScale
		startX := screenX - size/2
		startY := float32(screenHeight)/2 - size/2

		shade := 1 - tY/rayViewCells
		if shade < 0.2 {
			shade = 0.2
		}
		base := e.Color
		if e.HitTimer > 0 {
			base = color.RGBA{255, 255, 255, 255} // спалах після удару
		}
		col := color.RGBA{
			uint8(float32(base.R) * shade),
			uint8(float32(base.G) * shade),
			uint8(float32(base.B) * shade),
			255,
		}

		// Вертикальні смуги з тестом глибини проти стін.
		s0 := int(startX) / rayColW
		s1 := int(startX+size) / rayColW
		for strip := s0; strip <= s1; strip++ {
			if strip < 0 || strip >= len(zbuf) {
				continue
			}
			if tY >= zbuf[strip] {
				continue // за стіною → не малюємо
			}
			vector.FillRect(screen, float32(strip*rayColW), startY, rayColW, size, col, false)
		}
	}
}
