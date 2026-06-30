package main

import (
	"fmt"
	"image/color"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	etext "github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

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

// drawPixel малює квадрат з flash-ефектом, HP bar і міткою.
func drawPixel(screen *ebiten.Image, p Pixel) {
	// Flash: поки HitTimer > 0 — малюємо білим
	col := p.Color
	if p.HitTimer > 0 {
		col = color.RGBA{255, 255, 255, 255}
	}
	vector.FillRect(screen, p.X, p.Y, pixelSize, pixelSize, col, false)

	// HP bar — тільки для ворогів (MaxHP > 0)
	if p.MaxHP > 0 {
		const barH = 3
		barY := p.Y - barH - 1
		vector.FillRect(screen, p.X, barY, pixelSize, barH, color.RGBA{80, 0, 0, 200}, false)
		hpRatio := float32(p.HP) / float32(p.MaxHP)
		filled := float32(pixelSize) * hpRatio
		barColor := color.RGBA{
			R: uint8(255 * (1 - hpRatio)),
			G: uint8(220 * hpRatio),
			B: 0, A: 255,
		}
		vector.FillRect(screen, p.X, barY, filled, barH, barColor, false)
	}

	if p.Label != "" {
		cx := float64(p.X) + pixelSize/2
		cy := float64(p.Y) + pixelSize/2
		drawText(screen, p.Label, labelFontSize, cx, cy, color.RGBA{0, 0, 0, 255})
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
		vector.StrokeLine(screen, cx, cy, ex, ey, 1, col, false)
	}
	// Обрана дія — яскрава жовта стрілка.
	a := e.Brain.lastAction
	vector.StrokeLine(screen, cx, cy, cx+dirs8[a][0]*45, cy+dirs8[a][1]*45, 2,
		color.RGBA{255, 255, 0, 255}, false)
}

// Draw малює поточний стан на екрані.
func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{15, 15, 25, 255}) // темно-синій фон замість чистого чорного

	// Малюємо тайли рівня.
	// Стіни — темно-сірі, підлога — не малюється (фон і є підлогою).
	for row := 0; row < boidMapH; row++ {
		for col := 0; col < boidMapW; col++ {
			if tileMap[row][col] {
				x := float32(col * pixelSize)
				y := float32(row * pixelSize)
				vector.FillRect(screen, x, y, pixelSize, pixelSize, color.RGBA{55, 55, 75, 255}, false)
				// Тонкий контур стіни для об'єму
				vector.StrokeRect(screen, x, y, pixelSize, pixelSize, 1, color.RGBA{80, 80, 110, 255}, false)
			}
		}
	}

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
				vector.FillRect(screen, x, y, pixelSize, pixelSize, color.RGBA{255, 80, 0, uint8(a)}, false)
			}
		}
	}

	for _, e := range g.enemies {
		// Радіус огляду — дуже прозоре кільце навколо ворога
		cx := e.X + pixelSize/2
		cy := e.Y + pixelSize/2
		if showDetectionCircle {
			vector.StrokeCircle(screen, cx, cy, e.Cfg.DetectionRange, 1, color.RGBA{255, 255, 255, 5}, false)
		}
		if showWhiskers && e.Brain != nil {
			drawBrainSensors(screen, e, cx, cy)
		}
		drawPixel(screen, e)
	}
	drawPixel(screen, g.player)

	// Кільце атаки — StrokeCircle (контур) замість DrawFilledCircle (заповнене).
	// Виглядає чистіше як індикатор зони удару і не засліплює.
	// alpha = 180..0 — плавно зникає разом з attackTimer.
	if g.attackTimer > 0 {
		cx := g.player.X + pixelSize/2
		cy := g.player.Y + pixelSize/2
		alpha := uint8(180 * g.attackTimer / attackDuration)
		vector.StrokeCircle(screen, cx, cy, attackRadius, 2, color.RGBA{255, 255, 80, alpha}, true)
	}

	// HUD
	level := g.tick/levelUpEvery + 1
	playerSpeed := math.Sqrt(float64(g.player.VelX*g.player.VelX + g.player.VelY*g.player.VelY))
	white := color.RGBA{255, 255, 255, 255}
	cyan := color.RGBA{0, 220, 180, 255}
	fps := ebiten.ActualFPS()
	drawText(screen, fmt.Sprintf("LVL %d", level), 10, 30, 10, white)
	drawText(screen, fmt.Sprintf("SPD %.1f", playerSpeed), 10, screenWidth-32, 10, cyan)
	drawText(screen, fmt.Sprintf("DIF %.1f", g.difficulty), 10, screenWidth-32, 25, color.RGBA{255, 140, 50, 255})
	drawText(screen, fmt.Sprintf("FPS %.0f", fps), 10, screenWidth/2, 10, color.RGBA{150, 150, 150, 255})

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
}
