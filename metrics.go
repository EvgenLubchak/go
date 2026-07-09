package main

import (
	"fmt"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// ==========================================================================
// МЕТРИКИ СТЕНДУ — «міні-TensorBoard» прямо в грі (клавіша G).
//
// Навіщо: щойно ускладнимо мозок (локальне сприйняття, пам'ять), «дивитись і
// відчувати» перестане працювати — треба МІРЯТИ. Показуємо криву навчання
// (reward і TD-error у часі) + канарки Max Q та ε.
//
// Як читати:
//   reward ↑ → плато        — вчиться
//   TD-error ↓ → стабільний  — збігається
//   Max Q росте БЕЗМЕЖНО     — розбіжність (смертельна тріада!) → знижуй lr/γ
//   reward плаский           — не вчиться (масштаб reward / exploration / буфер)
// ==========================================================================

const (
	metricSamples = 240  // точок у кривій (ширина графіка)
	metricEvery   = 15   // семпл раз на N кадрів (не щокадру — шумно)
	metricEMA     = 0.08 // згладжування (менше = плавніше, повільніше реагує)
)

// curve — кільцевий буфер значень для лінійного графіка.
type curve struct {
	buf  [metricSamples]float32
	head int
	n    int
}

func (c *curve) push(v float32) {
	c.buf[c.head] = v
	c.head = (c.head + 1) % metricSamples
	if c.n < metricSamples {
		c.n++
	}
}

// at повертає i-те значення від найстарішого (0) до найновішого (n-1).
func (c *curve) at(i int) float32 {
	start := (c.head - c.n + metricSamples) % metricSamples
	return c.buf[(start+i)%metricSamples]
}

// Metrics збирає й зберігає показники навчання рою.
type Metrics struct {
	tick int

	reward float32 // згладжені поточні значення
	tdErr  float32
	maxQ   float32
	eps    float32
	inited bool

	rewardCurve curve
	tdCurve     curve

	// [ПАМʼЯТЬ] Кумулятивні лічильники ВІД МОМЕНТУ СКИДАННЯ (клавіша M). Скидаємо
	// вручну, коли рій уже навчився → міряємо саме навчену політику, а не історію.
	blindN      int // «сліпих рішень» (агент не бачив гравця, коли обирав дію)
	blindClosed int // ...із них скоротили дистанцію → ознака памʼяті
	catches     int // спіймань гравця
	window      int // кадрів від моменту скидання (для catch-rate у хв)
}

// resetCounters обнуляє кумулятивні лічильники заміру (клавіша M). Криві навчання
// НЕ чіпаємо — вони показують динаміку, а лічильники — підсумок навченого рою.
func (m *Metrics) resetCounters() {
	m.blindN, m.blindClosed, m.catches, m.window = 0, 0, 0, 0
}

// collect — раз/кадр (у Update, ПІСЛЯ trainBrains) збирає показники з мозків рою.
// Однопотоково → без гонок: lastReward писався в паралельній фазі (вже завершеній),
// акумулятори мережі — в trainBrains (теж завершеному).
func (m *Metrics) collect(g *Game) {
	var rSum float32
	var rN int
	var tdSum, qSum float32
	var tdN int
	seen := map[*Net]bool{}

	for i := range g.enemies {
		b := g.enemies[i].Brain
		if b == nil {
			continue
		}
		rSum += b.lastReward
		rN++
		// [ПАМʼЯТЬ] Забираємо «сліпі рішення» цього агента й скидаємо (однопотоково,
		// паралельна фаза calcAcceleration уже завершена → без гонок).
		m.blindN += b.mBlindN
		m.blindClosed += b.mBlindClosed
		b.mBlindN, b.mBlindClosed = 0, 0
		if b.net != nil && !seen[b.net] {
			seen[b.net] = true
			tdSum += b.net.mTDSum
			qSum += b.net.mQSum
			tdN += b.net.mTDN
			m.eps = b.epsilon() // репрезентативна ε
			b.net.mTDSum, b.net.mQSum, b.net.mTDN = 0, 0, 0
		}
	}

	var frameR, frameTD, frameQ float32
	if rN > 0 {
		frameR = rSum / float32(rN)
	}
	if tdN > 0 {
		frameTD = tdSum / float32(tdN)
		frameQ = qSum / float32(tdN)
	}

	if !m.inited {
		m.reward, m.tdErr, m.maxQ, m.inited = frameR, frameTD, frameQ, true
	} else {
		m.reward += (frameR - m.reward) * metricEMA
		m.tdErr += (frameTD - m.tdErr) * metricEMA
		m.maxQ += (frameQ - m.maxQ) * metricEMA
	}

	m.tick++
	m.window++ // [ПАМʼЯТЬ] кадрів від моменту скидання (для catch-rate)
	if m.tick%metricEvery == 0 {
		m.rewardCurve.push(m.reward)
		m.tdCurve.push(m.tdErr)
	}
}

// onoff — короткий підпис прапорця для ярлика конфігурації.
func onoff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// draw малює панель метрик (конфіг + числа + криві + памʼять/лов) у лівому куті.
func (m *Metrics) draw(screen *ebiten.Image) {
	const px, pw, ph = 12, 760, 180
	const py = screenHeight - ph - 12
	vector.FillRect(screen, px, py, pw, ph, color.RGBA{0, 0, 0, 170}, false)

	green := color.RGBA{90, 220, 120, 255}
	orange := color.RGBA{235, 150, 50, 255}
	gray := color.RGBA{170, 170, 180, 255}
	cyan := color.RGBA{95, 200, 220, 255}
	mem := color.RGBA{205, 140, 235, 255}   // blind-pursuit — метрика памʼяті
	yellow := color.RGBA{230, 215, 95, 255} // спіймання

	// Рядок 0: ЯРЛИК КОНФІГУРАЦІЇ — щоб скріншоти A/B самі себе документували.
	cfg := fmt.Sprintf("stk%d  local:%s  shared:%s  ai:%s",
		stackFrames, onoff(localSight), onoff(sharedBrain), onoff(aiPlayer))
	drawTextL(screen, cfg, 8, px+10, py+12, cyan)

	// Рядок 1: числа навчання.
	drawTextL(screen, fmt.Sprintf("reward %+.3f", m.reward), 9, px+10, py+28, green)
	drawTextL(screen, fmt.Sprintf("TD %.3f", m.tdErr), 9, px+150, py+28, orange)
	drawTextL(screen, fmt.Sprintf("maxQ %.2f", m.maxQ), 9, px+250, py+28, gray)

	// Крива навчання (кожна в СВОЄМУ масштабі — різні діапазони).
	gx, gy, gw, gh := float32(px+10), float32(py+40), float32(pw-20), float32(60)
	drawCurve(screen, &m.rewardCurve, gx, gy, gw, gh, green)
	drawCurve(screen, &m.tdCurve, gx, gy, gw, gh, orange)

	// Рядок 2: BLIND-PURSUIT — головна метрика памʼяті (частка «сліпих» кадрів,
	// де агент усе одно скоротив дистанцію). База ~50% (навмання) → вище = памʼять.
	bp := "—"
	if m.blindN > 0 {
		bp = fmt.Sprintf("%.0f%%", 100*float32(m.blindClosed)/float32(m.blindN))
	}
	drawTextL(screen, fmt.Sprintf("blind-chase %s  (n=%d)", bp, m.blindN), 9, px+10, py+ph-30, mem)

	// Рядок 3: спіймання + темп (catch/min) та ε; праворуч — легенда кривих.
	rate := float32(0)
	if m.window > 0 {
		rate = float32(m.catches) * (120 * 60) / float32(m.window) // TPS=120
	}
	drawTextL(screen, fmt.Sprintf("catch %d (%.1f/min)", m.catches, rate), 9, px+10, py+ph-14, yellow)
	drawTextL(screen, fmt.Sprintf("eps %.3f", m.eps), 8, px+185, py+ph-14, gray)
	drawTextL(screen, "reward", 8, px+250, py+ph-14, green)
	drawTextL(screen, "TD", 8, px+320, py+ph-14, orange)
}

// drawCurve малює полілінію значень, автомасштабуючи до прямокутника (x,y,w,h).
func drawCurve(screen *ebiten.Image, c *curve, x, y, w, h float32, col color.RGBA) {
	if c.n < 2 {
		return
	}
	lo, hi := c.at(0), c.at(0)
	for i := 1; i < c.n; i++ {
		v := c.at(i)
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if hi-lo < 1e-6 {
		hi = lo + 1 // рівна лінія — уникаємо ділення на нуль
	}

	var prevX, prevY float32
	for i := 0; i < c.n; i++ {
		v := c.at(i)
		nx := x + w*float32(i)/float32(c.n-1)
		ny := y + h - h*(v-lo)/(hi-lo) // більше значення → вище (менший y)
		if i > 0 {
			vector.StrokeLine(screen, prevX, prevY, nx, ny, 1.5, col, false)
		}
		prevX, prevY = nx, ny
	}
}
