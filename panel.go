package main

import (
	"fmt"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// ==========================================================================
// [ПАНЕЛЬ НАЛАШТУВАНЬ] Клавіша Tab. Перший крок виносу перемикачів із клавіш.
//
// КЕРУВАННЯ — ГІБРИД над ОДНИМ станом виділення (рішення обговорене й ухвалене):
//
//	миша/тачпад   рух курсора над рядком переносить вибір; клік = перемкнути.
//	              Тачпад мака для нас — звичайна миша: рух = курсор, тап = клік
//	              (glfw не віддає жестів). Ціль кліку — ВЕСЬ рядок, не чекбокс.
//	клавіатура    ↑↓ (та W/S) — вибір; Enter/Space/→ — вперед по значеннях;
//	              ← — назад. Руки не покидають клавіатуру під час замірів.
//
// Вибір переносить той, хто РУХАВСЯ ОСТАННІМ: стрілки не бʼються з нерухомим
// курсором (hover спрацьовує лише на зміну позиції курсора).
//
// Гра під відкритою панеллю НЕ спиняється — половина ручок тут це прилади
// порівняння («перемкнув AA — дивись на FPS тим самим оком»). Придушується лише
// ввід гравця (рух і ривок), бо стрілки віддані навігації. ESC закриває панель,
// а не гру (звичка «ESC = закрити модалку» не має коштувати сесії).
//
// Панель народжується з ДЕКЛАРАТИВНОЇ таблиці panelItems — того самого сорту
// «одне джерело істини», що й комірки стенду в одній параметризованій функції.
// persist позначає, чи пише зміна settings.json (лише «живі» ручки комфорту).
// ==========================================================================

var (
	panelOpen bool
	panelSel  int
	// Остання бачена позиція курсора: hover керує вибором лише коли курсор
	// РУХАЄТЬСЯ — інакше нерухома миша миттєво забирала б вибір у стрілок.
	panelCX, panelCY int
)

// panelItem — один рядок панелі. next/prev — крок значення вперед/назад
// (у перемикача обидва — той самий фліп).
type panelItem struct {
	group   string
	name    string
	value   func() string
	next    func()
	prev    func()
	persist bool // true → зміна пишеться в settings.json (лише ручки комфорту)
}

// cycleSS — крок драбиною суперсемплінгу. Логіка колишньої клавіші H, тепер
// двонаправлена; значення поза драбиною спершу стає на найближчий щабель.
func cycleSS(dir int) {
	renderScale = nearestSSStep(renderScale)
	i := 0
	for j, s := range ssLadder {
		if s == renderScale {
			i = j
			break
		}
	}
	renderScale = ssLadder[(i+dir+len(ssLadder))%len(ssLadder)]
}

// markDefault — значення з позначкою відхилення: панель, як і HUD зі
// settings.json, підсвічує лише НЕтипове.
func markDefault(cur, def string) string {
	if cur == def {
		return cur
	}
	return cur + "  (типово " + def + ")"
}

var panelItems = []panelItem{
	{
		group:   "Графіка",
		name:    "AA — згладжування шляхів",
		value:   func() string { return markDefault(onoff(antiAlias), onoff(antiAliasDefault)) },
		next:    func() { antiAlias = !antiAlias; saveSettings() },
		prev:    func() { antiAlias = !antiAlias; saveSettings() },
		persist: true,
	},
	{
		group: "Графіка",
		name:  "SS — суперсемплінг",
		value: func() string {
			return markDefault(fmt.Sprintf("%g×", renderScale), fmt.Sprintf("%g×", float32(renderScaleDefault)))
		},
		next:    func() { cycleSS(+1); saveSettings() },
		prev:    func() { cycleSS(-1); saveSettings() },
		persist: true,
	},
}

// Геометрія панелі — ОДНА функція для малювання й хіт-тесту (розійтись не мають
// права: рядок, який малюється не там, де клікається, — то зламана панель).
const (
	panelX      = 12
	panelY      = 60
	panelW      = 560
	panelRowH   = 34
	panelPad    = 14
	panelHeadH  = 30 // рядок заголовка групи
	panelFootH  = 26 // рядок підказки керування
	panelFontSz = 15
)

// panelRowRect — екранний прямокутник рядка i (лише пункти, без заголовка).
func panelRowRect(i int) (x0, y0, x1, y1 float32) {
	x0 = panelX + panelPad/2
	y0 = float32(panelY + panelPad + panelHeadH + i*panelRowH)
	return x0, y0, x0 + panelW - panelPad, y0 + panelRowH
}

// panelRowAt — індекс рядка під точкою (−1 = мимо).
func panelRowAt(cx, cy int) int {
	for i := range panelItems {
		x0, y0, x1, y1 := panelRowRect(i)
		if float32(cx) >= x0 && float32(cx) < x1 && float32(cy) >= y0 && float32(cy) < y1 {
			return i
		}
	}
	return -1
}

// handlePanelInput — увесь ввід відкритої панелі (клавіатура + миша/тачпад).
// Викликається з Update лише коли panelOpen.
func handlePanelInput() {
	n := len(panelItems)
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowUp) || inpututil.IsKeyJustPressed(ebiten.KeyW) {
		panelSel = (panelSel - 1 + n) % n
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowDown) || inpututil.IsKeyJustPressed(ebiten.KeyS) {
		panelSel = (panelSel + 1) % n
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowLeft) || inpututil.IsKeyJustPressed(ebiten.KeyA) {
		panelItems[panelSel].prev()
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowRight) || inpututil.IsKeyJustPressed(ebiten.KeyD) ||
		inpututil.IsKeyJustPressed(ebiten.KeyEnter) || inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		panelItems[panelSel].next()
	}

	cx, cy := ebiten.CursorPosition()
	if cx != panelCX || cy != panelCY {
		panelCX, panelCY = cx, cy
		if row := panelRowAt(cx, cy); row >= 0 {
			panelSel = row
		}
	}
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		if row := panelRowAt(cx, cy); row >= 0 {
			panelSel = row
			panelItems[row].next()
		}
	}
}

// drawPanel — малює панель поверх усього (HUD-шар, рідна роздільність).
func drawPanel(screen *ebiten.Image) {
	h := float32(panelPad*2 + panelHeadH + len(panelItems)*panelRowH + panelFootH)
	vector.FillRect(screen, panelX, panelY, panelW, h, color.RGBA{0, 0, 0, 210}, false)

	amber := color.RGBA{240, 200, 90, 255}
	dim := color.RGBA{150, 160, 175, 255}
	white := color.RGBA{230, 235, 240, 255}

	group := ""
	y := float64(panelY + panelPad + panelHeadH - 10)
	drawTextL(screen, "НАЛАШТУВАННЯ", panelFontSz, panelX+panelPad, y-4, dim)

	for i, it := range panelItems {
		x0, y0, x1, y1 := panelRowRect(i)
		if it.group != group {
			group = it.group
			// Заголовок групи малюємо В рядку першого пункту групи справа —
			// v1 має одну групу, тож окремих рядків-заголовків поки не заводимо.
			drawTextL(screen, "["+group+"]", panelFontSz*0.85, float64(x1)-110,
				float64(y0)+float64(panelRowH)*0.62, dim)
		}
		if i == panelSel {
			vector.FillRect(screen, x0, y0, x1-x0, y1-y0, color.RGBA{50, 75, 95, 255}, false)
		}
		col := white
		if i == panelSel {
			col = amber
		}
		ty := float64(y0) + float64(panelRowH)*0.62
		drawTextL(screen, it.name, panelFontSz, float64(x0)+8, ty, col)
		drawTextL(screen, it.value(), panelFontSz, float64(x0)+300, ty, col)
	}

	fy := float64(panelY) + float64(h) - panelPad + 2
	drawTextL(screen, "Tab/Esc закрити   ↑↓ вибір   Enter/клік перемкнути   ←→ крок",
		panelFontSz*0.8, panelX+panelPad, fy, dim)
}
