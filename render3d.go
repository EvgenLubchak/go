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
	rayFOV = math.Pi / 3 // поле зору ~60°
	// Ширина смуги в пікселях. 1 = піксельна чіткість країв замість сходинок.
	//
	// Коштує це менше, ніж здається: DDA долітає до стіни за ~10 кроків, тож 1700
	// променів це ~17 тисяч операцій за кадр — на тлі 60 юнітів із ворсом дрібниця.
	// Дорогими стали б СПРАЙТИ (один прямокутник на колонку), тому drawSprites малює їх
	// ПАКЕТАМИ суміжних видимих колонок, а не по одній.
	rayColW      = 1
	rayMaxCell   = 80   // стеля кроків DDA (щоб не зациклитись)
	rayViewCells = 26.0 // дальність для затемнення (клітинки)
	spriteScale  = 0.8  // розмір ворога-спрайта відносно стіни-клітинки
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
	drawSkyAndFloor(screen)

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

		// [ТУМАН] Далі → ближче до кольору ГОРИЗОНТУ, а не до чорного. Затемнення в
		// нуль читається як «темно», а розчинення в горизонті — як «далеко»; це і є
		// атмосферна перспектива, і коштує вона рівно стільки ж.
		shade := 1 - perpDist/rayViewCells
		if shade < 0 {
			shade = 0
		}
		if side == 1 {
			shade *= 0.72 // горизонтальна грань темніша → ребра стін читаються
		}
		near := color.RGBA{150, 150, 195, 255} // звичайна стіна зблизька
		if boundary {
			near = color.RGBA{205, 130, 45, 255} // межа рівня — тепла бурштинова
		}
		col := fogMix(near, shade)
		vector.FillRect(screen, float32(c*rayColW), y0, rayColW, lineH, col, false)
	}

	g.drawSprites(screen, zbuf[:], posX, posY, dirX, dirY, planeX, planeY)
}

// drawSkyAndFloor — градієнт замість двох плоских заливок.
//
// Той самий прийом, що в drawSea: рівний колір читається як «пофарбоване тло», а перехід
// — як ГЛИБИНА. Обидва градієнти сходяться до кольору ГОРИЗОНТУ, у якому розчиняються й
// далекі стіни (fogMix), тож сцена змикається в одну атмосферу замість трьох окремих смуг.
func drawSkyAndFloor(screen *ebiten.Image) {
	h := float32(screenHeight) / 2 / rayBands
	for i := 0; i < rayBands; i++ {
		t := float32(i) / (rayBands - 1) // 0 = зеніт, 1 = горизонт
		sky := color.RGBA{
			uint8(float32(skyTopR) + (fogR-skyTopR)*t),
			uint8(float32(skyTopG) + (fogG-skyTopG)*t),
			uint8(float32(skyTopB) + (fogB-skyTopB)*t), 255,
		}
		vector.FillRect(screen, 0, float32(i)*h, screenWidth, h+1, sky, false)

		// Підлога дзеркально: від горизонту вниз, до ближчого й темнішого.
		floor := color.RGBA{
			uint8(float32(floorFarR) + (floorNearR-floorFarR)*t),
			uint8(float32(floorFarG) + (floorNearG-floorFarG)*t),
			uint8(float32(floorFarB) + (floorNearB-floorFarB)*t), 255,
		}
		vector.FillRect(screen, 0, screenHeight/2+float32(i)*h, screenWidth, h+1, floor, false)
	}
}

// fogMix — колір на відстані: від кольору горизонту (shade 0) до near (shade 1).
func fogMix(near color.RGBA, shade float32) color.RGBA {
	if shade > 1 {
		shade = 1
	}
	return color.RGBA{
		uint8(float32(fogR) + (float32(near.R)-fogR)*shade),
		uint8(float32(fogG) + (float32(near.G)-fogG)*shade),
		uint8(float32(fogB) + (float32(near.B)-fogB)*shade),
		255,
	}
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
		if shade < 0 {
			shade = 0
		}
		base := e.Color
		if e.HitTimer > 0 {
			base = color.RGBA{255, 255, 255, 255} // спалах після удару
		}

		// [ПАКЕТУВАННЯ] Один прямокутник на КОЛОНКУ був би 400 викликів на близького
		// юніта при rayColW = 1. Замість цього йдемо колонками й накопичуємо СУМІЖНІ
		// видимі — малюємо кожен такий пробіг одним разом. Видимість міняється лише на
		// краях стін, тож пробігів зазвичай один-два.
		s0 := int(startX) / rayColW
		s1 := int(startX+size) / rayColW
		runStart := -1
		flush := func(from, to int) {
			if from < 0 {
				return
			}
			x := float32(from * rayColW)
			w := float32((to - from + 1) * rayColW)
			drawSpriteSlab(screen, x, startY, w, size, base, shade, e)
		}
		for strip := s0; strip <= s1; strip++ {
			vis := strip >= 0 && strip < len(zbuf) && tY < zbuf[strip]
			switch {
			case vis && runStart < 0:
				runStart = strip
			case !vis && runStart >= 0:
				flush(runStart, strip-1)
				runStart = -1
			}
		}
		flush(runStart, s1)
	}
}

// drawSpriteSlab — шматок спрайта: вертикальний градієнт плюс смужка HP згори.
//
// Градієнт (світліше згори, темніше знизу) коштує spriteBands прямокутників на пробіг і
// прибирає головну ваду 3D-виду: юніт перестає бути ПЛИТКОЮ кольору й отримує обʼєм.
// Смужка HP — той самий сигнал, що у виді зверху, тільки тут він потрібніший: у 3D не
// видно ні ворсу, ні форми тіла, тобто стану юніта не прочитати ніяк інакше.
func drawSpriteSlab(screen *ebiten.Image, x, y, w, h float32, base color.RGBA, shade float32, e *Pixel) {
	bh := h / spriteBands
	for b := 0; b < spriteBands; b++ {
		t := float32(b) / (spriteBands - 1) // 0 = верх, 1 = низ
		k := spriteTopMul + (spriteBotMul-spriteTopMul)*t
		lit := color.RGBA{
			uint8(min32(float32(base.R)*k, 255)),
			uint8(min32(float32(base.G)*k, 255)),
			uint8(min32(float32(base.B)*k, 255)), 255,
		}
		vector.FillRect(screen, x, y+float32(b)*bh, w, bh+1, fogMix(lit, shade), false)
	}

	if e.MaxHP <= 0 {
		return
	}
	frac := float32(e.HP) / float32(e.MaxHP)
	if frac < 0 {
		frac = 0
	}
	hpH := h * spriteHPH
	vector.FillRect(screen, x, y-hpH*1.6, w, hpH, fogMix(color.RGBA{40, 40, 50, 255}, shade), false)
	vector.FillRect(screen, x, y-hpH*1.6, w*frac, hpH, fogMix(color.RGBA{80, 230, 90, 255}, shade), false)
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
