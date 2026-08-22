package main

import (
	"fmt"
	"image/color"
	"log"

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
//	клавіатура    ↑↓ (та W/S) — вибір; Enter/→ — вперед по значеннях; ← — назад.
//	              Space і A/D дій НЕ мають: гра під панеллю живе, і рефлекси
//	              WASD/ривка не повинні перемикати налаштування (перевірено
//	              першим же дебагом — див. handlePanelInput).
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
	next    func(g *Game)
	prev    func(g *Game)
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

// toggleSound — вмикає/вимикає фоновий ритм. Увімкнення грає ПОТОЧНИЙ патерн
// (startBeat без просування — інакше кожен фліп звуку крутив би плейлист).
// g може бути nil у тестах — тоді аудіо не чіпаємо, лише прапорець і файл.
func toggleSound(g *Game) {
	soundEnabled = !soundEnabled
	if g != nil {
		if soundEnabled {
			startBeat(g.difficulty)
		} else {
			stopBeat()
		}
	}
	saveSettings()
}

// cyclePattern — інший ритм у циклі, в обидва боки (замінив клавішу B, яка вміла
// лише вперед і мовчала при вимкненому звуці). Вибір працює й БЕЗ звуку — патерн
// просто заграє, коли звук увімкнуть.
func cyclePattern(g *Game, dir int) {
	n := len(patterns)
	currentPatternIdx = ((currentPatternIdx+dir)%n + n) % n
	if g != nil && soundEnabled {
		startBeat(g.difficulty)
	}
}

// toggleTPS — 60 ↔ 120 (логіка колишньої клавіші T). Усе в грі рахується в
// КАДРАХ, тож це рівномірне сповільнення всього одразу; біт переганяємо, бо він
// єдиний живе в реальних секундах (див. gameTPS у main.go). g може бути nil у
// тестах — тоді біт не чіпаємо (звук і так вимкнений поза грою).
func toggleTPS(g *Game) {
	if gameTPS == 120 {
		gameTPS = 60
	} else {
		gameTPS = 120
	}
	ebiten.SetTPS(gameTPS)
	if g != nil {
		startBeat(g.difficulty)
	}
	saveSettings()
}

var panelItems = []panelItem{
	{
		group:   "Ігрові налаштування",
		name:    "AA — згладжування шляхів",
		value:   func() string { return markDefault(onoff(antiAlias), onoff(antiAliasDefault)) },
		next:    func(_ *Game) { antiAlias = !antiAlias; saveSettings() },
		prev:    func(_ *Game) { antiAlias = !antiAlias; saveSettings() },
		persist: true,
	},
	{
		group: "Ігрові налаштування",
		name:  "SS — суперсемплінг",
		value: func() string {
			return markDefault(fmt.Sprintf("%g×", renderScale), fmt.Sprintf("%g×", float32(renderScaleDefault)))
		},
		next:    func(_ *Game) { cycleSS(+1); saveSettings() },
		prev:    func(_ *Game) { cycleSS(-1); saveSettings() },
		persist: true,
	},
	{
		group:   "Ігрові налаштування",
		name:    "Звук — фоновий ритм",
		value:   func() string { return markDefault(onoff(soundEnabled), onoff(soundEnabledDefault)) },
		next:    toggleSound,
		prev:    toggleSound,
		persist: true,
	},
	{
		group: "Ігрові налаштування",
		name:  "Ритм — барабанний патерн",
		value: func() string {
			n := len(patterns)
			name := patterns[((currentPatternIdx%n)+n)%n].name
			if !soundEnabled {
				return name + "  (звук вимк)"
			}
			return name
		},
		next: func(g *Game) { cyclePattern(g, +1) },
		prev: func(g *Game) { cyclePattern(g, -1) },
		// НЕ персиститься: патерн — стан сесії (скидається рестартом, чергується
		// щорівня), а не налаштування комфорту.
		persist: false,
	},
	{
		group: "Ігрові налаштування",
		name:  "Flow-field — шар поля",
		value: func() string {
			names := [3]string{"вимк", "до сторони гравця", "до ворогів"}
			return markDefault(names[showFlowField%3], names[0])
		},
		next: func(_ *Game) { showFlowField = (showFlowField + 1) % 3 },
		prev: func(_ *Game) { showFlowField = (showFlowField + 2) % 3 },
		// НЕ персиститься: діагностичний шар — стан сесії, як ритм.
		persist: false,
	},
	{
		group: "Ігрові налаштування",
		name:  "TPS — темп симуляції",
		value: func() string {
			return markDefault(fmt.Sprintf("%d", gameTPS), fmt.Sprintf("%d", gameTPSDefault))
		},
		next:    toggleTPS,
		prev:    toggleTPS,
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

// panelAct — ЄДИНА точка застосування дії панелі, зі СЛІДОМ У КОНСОЛІ: перший
// же дебаг показав, що без сліду неможливо відрізнити «одна дія спрацювала
// двічі» від «прилетіли дві дії» (подвійний тап тачпада, рефлекторна клавіша).
// Кожна зміна — один рядок: джерело, пункт, старе → нове.
func panelAct(g *Game, i, dir int, src string) {
	it := &panelItems[i]
	before := it.value()
	if dir < 0 {
		it.prev(g)
	} else {
		it.next(g)
	}
	log.Printf("панель[%s] %s: %s → %s", src, it.name, before, it.value())
}

// handlePanelInput — увесь ввід відкритої панелі (клавіатура + миша/тачпад).
// Викликається з Update лише коли panelOpen.
//
// ⚠️ ДІЇ — ЛИШЕ Enter, ←/→ і клік. Space та A/D зумисно ПРИБРАНІ після першого
// ж дебагу: гра під панеллю живе, і пальці за звичкою продовжують «грати» —
// рефлекторний Space (це ж ривок!) чи D (рух!) непомітно перемикали вибраний
// рядок. Так «максимальний пресет» AA + SS×4 і зʼявлявся нізвідки. W/S у
// навігації лишаються: вибір без зміни значення — нешкідливий рефлекс.
func handlePanelInput(g *Game) {
	n := len(panelItems)
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowUp) || inpututil.IsKeyJustPressed(ebiten.KeyW) {
		panelSel = (panelSel - 1 + n) % n
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowDown) || inpututil.IsKeyJustPressed(ebiten.KeyS) {
		panelSel = (panelSel + 1) % n
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowLeft) {
		panelAct(g, panelSel, -1, "←")
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowRight) {
		panelAct(g, panelSel, +1, "→")
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyEnter) {
		panelAct(g, panelSel, +1, "Enter")
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
			panelAct(g, row, +1, "клік")
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

	y := float64(panelY + panelPad + panelHeadH - 10)
	drawTextL(screen, "НАЛАШТУВАННЯ", panelFontSz, panelX+panelPad, y-4, dim)
	// Назва групи — у рядку заголовка справа: v1 має ОДНУ групу, і довге
	// «[Ігрові налаштування]» в рядку пункту билось би зі стовпчиком значень.
	// Кілька груп вимагатимуть власних рядків-заголовків і нової геометрії.
	drawTextL(screen, "["+panelItems[0].group+"]", panelFontSz*0.85,
		float64(panelX+panelW)-225, y-4, dim)

	for i, it := range panelItems {
		x0, y0, x1, y1 := panelRowRect(i)
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
