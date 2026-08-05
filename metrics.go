package main

import (
	"fmt"
	"image/color"
	"math"
	"strings"

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
	metricSamples = 480  // точок у кривій (ширина графіка ≈ 60с історії)
	metricEvery   = 15   // семпл раз на N кадрів (не щокадру — шумно)
	metricEMA     = 0.08 // згладжування (менше = плавніше, повільніше реагує)

	// [ПАНЕЛЬ] Один масштаб на всю панель: ширина, шрифти, висота рядків і графіка
	// рахуються від нього. Хочеш більшу/меншу панель — крути лише це число.
	metricScale = 2.2

	// Базові розміри (при metricScale = 1).
	metricBaseW      = 380 // ширина панелі
	metricBasePad    = 10  // внутрішні відступи
	metricBaseRowH   = 14  // висота рядка вулика
	metricBaseGraphH = 80  // висота області графіка
	metricBaseFont   = 9   // кегль тексту

	// [ПАМʼЯТЬ] Мінімум сліпих кадрів, щоб агента врахувати в по-агентному
	// відсотку. Без порогу юніт із трьома кадрами дав би 0% або 100% і смикав
	// би середнє нарівні з тим, хто набрав тисячі.
	blindAgentMin = 100
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

// hiveStat — показники ОДНОГО вулика (типу мозку). Раніше метрики усереднювались
// по ВСІХ мозках разом, і це вже брехало: у вбивці нагорода іншого масштабу (+5 за
// вбивство проти ±0.5 у рою), тож середнє змішувало різні речі. З трьома командами
// стало б зовсім нечитабельно.
type hiveStat struct {
	label   string     // «brain» / «killer» / «ally» — з імені файлу ваг
	mem     string     // контракт памʼяті ЦЬОГО вулика: «gru» або «stk4/10»
	horizon string     // «γ0.99» — ЛИШЕ якщо відрізняється від глобального дефолту
	color   color.RGBA // колір юнітів цього типу → лінія збігається з тим, що на полі
	reward  float32    // згладжені (EMA) поточні значення
	tdErr   float32
	maxQ    float32
	eps     float32
	inited  bool
	curve   curve // крива reward саме цього вулика
}

// Metrics збирає й зберігає показники навчання — ОКРЕМО по кожному вулику.
type Metrics struct {
	tick int

	hives map[string]*hiveStat // ключ — файл ваг (= тип мозку)
	order []string             // стабільний порядок показу (як зустріли)

	// [ПАМʼЯТЬ] Кумулятивні лічильники ВІД МОМЕНТУ СКИДАННЯ (клавіша M). Скидаємо
	// вручну, коли рій уже навчився → міряємо саме навчену політику, а не історію.
	blindN      int // «сліпих рішень» (агент не бачив гравця, коли обирав дію)
	blindClosed int // ...із них скоротили дистанцію → ознака памʼяті
	catches     int // спіймань гравця
	window      int // кадрів від моменту скидання (для catch-rate у хв)

	// Знаменник для ЧАСТКИ часу наосліп: сума «агент × кадр» за період. Рахуємо
	// саме так, а не як window×8, бо юніти можуть гинути — інакше після смерті
	// частка занижувалась би без жодної зміни в поведінці.
	unitFrames int

	// [ПАМʼЯТЬ] Ті самі сліпі рішення, але НАРІЗНО по агентах — див. blindPerAgent.
	agents map[*Brain]*blindAgent
}

// blindAgent — сліпі рішення ОДНОГО агента за період заміру.
//
// Навіщо окремо від blindN/blindClosed: котловий відсоток зважений ПО КАДРАХ, а
// кадри розподілені між агентами вкрай нерівно. Юніт, що знайшов ціль і висить
// біля неї, перестає давати сліпі кадри взагалі; юніт, застряглий за текстурою,
// дає рівно один за тік до кінця прогону — і майже всі невдалі. За пʼять хвилин
// при 120 TPS це ≈36000 кадрів з одного невдахи проти вибірки у 18000. Тобто
// котловий відсоток здатен на дві третини складатися з одного застряглого юніта,
// а скільки їх застрягне — випадковість прогону. Звідси і розкид 58/66 між
// двома прогонами того самого конфігу.
type blindAgent struct {
	n      int // сліпих рішень цього агента
	closed int // ...із них із прогресом до цілі
}

// resetCounters обнуляє кумулятивні лічильники заміру (клавіша M). Криві навчання
// НЕ чіпаємо — вони показують динаміку, а лічильники — підсумок навченого рою.
func (m *Metrics) resetCounters() {
	m.blindN, m.blindClosed, m.catches, m.window = 0, 0, 0, 0
	m.unitFrames = 0
	m.agents = nil
}

// collect — раз/кадр (у Update, ПІСЛЯ trainBrains) збирає показники з мозків рою.
// Однопотоково → без гонок: lastReward писався в паралельній фазі (вже завершеній),
// акумулятори мережі — в trainBrains (теж завершеному).
func (m *Metrics) collect(g *Game) {
	if m.hives == nil {
		m.hives = map[string]*hiveStat{}
	}
	// Сума нагород і кількість агентів — ОКРЕМО по кожному вулику.
	type acc struct {
		rSum float32
		rN   int
	}
	sums := map[string]*acc{}
	seen := map[*Net]bool{}

	for i := range g.units {
		u := &g.units[i]
		b := u.Brain
		if b == nil || b.net == nil {
			continue
		}
		key := b.net.file
		if key == "" {
			key = "ephemeral" // мозок-жертва в self-play (не зберігається)
		}
		h := m.hives[key]
		if h == nil {
			h = &hiveStat{label: hiveLabel(key), mem: b.net.mem.label(), color: u.Cfg.Color}
			// Показуємо γ лише коли вона НЕ дефолтна: панель має підсвічувати
			// незвичайне, а не повторювати те саме пʼять разів.
			if math.Abs(float64(b.net.gamma-qGamma)) > 1e-6 {
				h.horizon = fmt.Sprintf("γ%.3g", b.net.gamma)
			}
			m.hives[key] = h
			m.order = append(m.order, key)
		}
		if sums[key] == nil {
			sums[key] = &acc{}
		}
		sums[key].rSum += b.lastReward
		sums[key].rN++
		h.eps = b.epsilon()

		// [ПАМʼЯТЬ] Забираємо «сліпі рішення» агента й скидаємо (однопотоково,
		// паралельна фаза calcAcceleration уже завершена → без гонок).
		m.unitFrames++ // знаменник частки наосліп: цей агент прожив цей кадр
		if b.mBlindN > 0 {
			if m.agents == nil {
				m.agents = map[*Brain]*blindAgent{}
			}
			a := m.agents[b]
			if a == nil {
				a = &blindAgent{}
				m.agents[b] = a
			}
			a.n += b.mBlindN
			a.closed += b.mBlindClosed
		}
		m.blindN += b.mBlindN
		m.blindClosed += b.mBlindClosed
		b.mBlindN, b.mBlindClosed = 0, 0

		// TD/maxQ — з МЕРЕЖІ, тож беремо раз на унікальну мережу.
		if !seen[b.net] {
			seen[b.net] = true
			if n := b.net.mTDN; n > 0 {
				frameTD := b.net.mTDSum / float32(n)
				frameQ := b.net.mQSum / float32(n)
				if !h.inited {
					h.tdErr, h.maxQ = frameTD, frameQ
				} else {
					h.tdErr += (frameTD - h.tdErr) * metricEMA
					h.maxQ += (frameQ - h.maxQ) * metricEMA
				}
			}
			b.net.mTDSum, b.net.mQSum, b.net.mTDN = 0, 0, 0
		}
	}

	for key, a := range sums {
		h := m.hives[key]
		var frameR float32
		if a.rN > 0 {
			frameR = a.rSum / float32(a.rN)
		}
		if !h.inited {
			h.reward, h.inited = frameR, true
		} else {
			h.reward += (frameR - h.reward) * metricEMA
		}
	}

	m.tick++
	m.window++ // [ПАМʼЯТЬ] кадрів від моменту скидання (для catch-rate)
	if m.tick%metricEvery == 0 {
		for _, key := range m.order {
			m.hives[key].curve.push(m.hives[key].reward)
		}
	}
}

// blindPerAgent — відсоток «сліпих із прогресом», усереднений ПО АГЕНТАХ, а не
// по кадрах. Кожен агент важить однаково, скільки б кадрів не набрав, тож один
// застряглий за текстурою більше не визначає підсумок за всіх.
//
// Повертає ще й межі lo/hi — саме вони роблять забруднення видимим: якщо семеро
// дають 70–85%, а восьмий 4%, то восьмий стоїть у стіні, і це видно з панелі, а
// не з здогадок після прогону.
//
// Ключі — вказівники на Brain; мертві юніти лишаються в мапі до кінця замІру. Так і
// треба: їхні сліпі рішення були реальними, викидати їх заднім числом означало б
// підганяти вибірку під тих, хто дожив.
//
// АЛЕ рестарт (клавіша R) мапу скидає — це вже інший СВІТ, і усереднювати по агентах
// із різних світів безглуздо. Без цього скидання лічильник «ag» показував кількість
// життів, а не юнітів: один стражник після трьох смертей читався як «4 ag».
func (m *Metrics) blindPerAgent() (mean, lo, hi float32, agents int) {
	lo = 1
	for _, a := range m.agents {
		if a.n < blindAgentMin {
			continue
		}
		r := float32(a.closed) / float32(a.n)
		mean += r
		if r < lo {
			lo = r
		}
		if r > hi {
			hi = r
		}
		agents++
	}
	if agents == 0 {
		return 0, 0, 0, 0
	}
	return mean / float32(agents), lo, hi, agents
}

// hiveLabel — коротка назва вулика з імені файлу ваг: "killer_weights.json" → "killer".
func hiveLabel(file string) string {
	name := strings.TrimSuffix(file, ".json")
	return strings.TrimSuffix(name, "_weights")
}

// onoff — короткий підпис прапорця для ярлика конфігурації.
func onoff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// draw малює панель метрик: рядок на КОЖЕН вулик + криві його кольором.
// Усі розміри — від metricScale, тож панель масштабується одним числом.
func (m *Metrics) draw(screen *ebiten.Image) {
	const (
		pad    = metricBasePad * metricScale
		pw     = metricBaseW * metricScale
		rowH   = metricBaseRowH * metricScale
		graphH = metricBaseGraphH * metricScale
		font   = metricBaseFont * metricScale
	)
	ph := float32(pad*2+graphH+rowH) + float32(rowH)*float32(len(m.order)+2) // +1 рядок конфігу, +2 рядки підсумків
	px := float32(12)
	py := float32(screenHeight) - ph - 12
	vector.FillRect(screen, px, py, float32(pw), ph, color.RGBA{0, 0, 0, 190}, false)

	cyan := color.RGBA{95, 200, 220, 255}
	mem := color.RGBA{205, 140, 235, 255}
	yellow := color.RGBA{230, 215, 95, 255}

	// Рядок 0: ЯРЛИК КОНФІГУРАЦІЇ — щоб скріншоти A/B самі себе документували.
	y := float64(py) + pad + rowH*0.7
	// stk<кадрів>/<крок>: САМЕ КРОК визначає, чи пам'ять суцільна, чи дірчаста, —
	// а без нього два різні конфіги дають однаковий ярлик (stk4 при кроці 60 і
	// при кроці 10). Один раз уже звіряли скріни навгад; більше не треба.
	// frz — чи заморожена політика. Це найважливіший прапорець на скріні: замір із
	// frz:off і frz:on відповідають на РІЗНІ питання, і сплутати їх не можна.
	//
	// Контракту памʼяті тут БІЛЬШЕ НЕМА: він переїхав у рядок кожного вулика, бо став
	// властивістю мережі. Раніше «mem:stk stk4/10» малювалось раз на всю панель — і це
	// був видимий слід глобального прапорця: різні типи фізично не могли мати різну
	// памʼять. Тут лишилось тільки справді спільне для всіх.
	cfgCol := cyan
	if frozenPolicy {
		cfgCol = color.RGBA{120, 235, 140, 255} // заморожено → зелений, видно здалеку
	}
	drawTextL(screen, fmt.Sprintf("local:%s  shared:%s  ai:%s  frz:%s  eps %.3f",
		onoff(localSight), onoff(sharedBrain), onoff(aiPlayer),
		onoff(frozenPolicy), m.firstEps()), font*0.85, float64(px)+pad, y, cfgCol)

	// Рядок на КОЖЕН вулик — свої reward/TD/maxQ, кольором своїх юнітів.
	for _, key := range m.order {
		h := m.hives[key]
		y += rowH
		drawTextL(screen, fmt.Sprintf("%-7s %-9s %-6s r %+.3f  TD %.3f  Q %.2f",
			h.label, h.mem, h.horizon, h.reward, h.tdErr, h.maxQ), font, float64(px)+pad, y, h.color)
	}

	// Криві reward — по одній на вулик, тим самим кольором, у СПІЛЬНОМУ масштабі
	// (щоб вулики можна було порівнювати між собою, а не кожен у своїй системі).
	gx := px + float32(pad)
	gy := float32(y) + float32(rowH)*0.6
	gw := float32(pw - pad*2)
	lo, hi := m.curveRange()
	// Нульова лінія — орієнтир «вчиться / деградує».
	if lo < 0 && hi > 0 {
		zy := gy + float32(graphH) - float32(graphH)*(0-lo)/(hi-lo)
		vector.StrokeLine(screen, gx, zy, gx+gw, zy, 1, color.RGBA{110, 110, 130, 140}, false)
	}
	for _, key := range m.order {
		drawCurveIn(screen, &m.hives[key].curve, gx, gy, gw, float32(graphH), lo, hi, m.hives[key].color)
	}
	drawTextL(screen, fmt.Sprintf("%+.2f", hi), font*0.8, float64(gx), float64(gy)+7, color.RGBA{130, 130, 150, 200})
	drawTextL(screen, fmt.Sprintf("%+.2f", lo), font*0.8, float64(gx), float64(gy)+graphH-7, color.RGBA{130, 130, 150, 200})

	// Підсумкові лічильники заміру (скидаються клавішею M).
	bp := "—"
	if m.blindN > 0 {
		bp = fmt.Sprintf("%.0f%%", 100*float32(m.blindClosed)/float32(m.blindN))
	}
	rate := float32(0)
	if m.window > 0 {
		rate = float32(m.catches) * (120 * 60) / float32(m.window) // TPS=120
	}
	fy := float64(py+ph) - pad - rowH*0.2
	// [ГОЛОВНА МЕТРИКА] Частка часу, коли агент НЕ бачив ціль.
	//
	// Вимірювання показало, що blind-chase — метрика УМОВНА: вона рахує лише
	// сліпі кадри, тож конфіг, який майже не губить ціль, оцінюється по жменьці
	// вироджених кадрів (мікрозатемнення в товкотнечі впритул) і виглядає
	// посередньо. Частка ж наосліп безумовна й міряє саме здатність утримувати
	// ціль. На стенді вона розвела конфіги в шість разів там, де blind-chase
	// показував різницю в межах шуму.
	bf := "—"
	if m.unitFrames > 0 {
		bf = fmt.Sprintf("%.1f%%", 100*float32(m.blindN)/float32(m.unitFrames))
	}
	drawTextL(screen, fmt.Sprintf("blind %s   chase %s (n=%d)", bf, bp, m.blindN), font, float64(px)+pad, fy-rowH, mem)
	drawTextL(screen, fmt.Sprintf("catch %d (%.1f/min)", m.catches, rate), font, float64(px)+pw*0.5, fy-rowH, yellow)

	// Другий рядок — ТОЙ САМИЙ показник, але зважений по агентах, і межі розкиду.
	// Верхній рядок лишаємо, щоб уже зняті прогони мали з чим порівнюватись.
	pa := "per-agent —"
	if mean, lo, hi, k := m.blindPerAgent(); k > 0 {
		pa = fmt.Sprintf("per-agent %.0f%%  (%d ag  %.0f..%.0f%%)", 100*mean, k, 100*lo, 100*hi)
	}
	drawTextL(screen, pa, font, float64(px)+pad, fy, color.RGBA{170, 120, 220, 255})
}

// firstEps — ε будь-якого вулика (у всіх однакова формула, показуємо як довідку).
func (m *Metrics) firstEps() float32 {
	for _, key := range m.order {
		return m.hives[key].eps
	}
	return 0
}

// curveRange — СПІЛЬНИЙ масштаб для всіх кривих, щоб вулики можна було
// порівнювати між собою (у кожного свій масштаб — це були б різні системи координат).
func (m *Metrics) curveRange() (lo, hi float32) {
	lo, hi = 1e30, -1e30
	for _, key := range m.order {
		c := &m.hives[key].curve
		for i := 0; i < c.n; i++ {
			v := c.at(i)
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
	}
	if lo > hi {
		return 0, 1
	}
	if hi-lo < 1e-6 {
		hi = lo + 1
	}
	return lo, hi
}

// drawCurveIn малює полілінію в ЗАДАНОМУ масштабі [lo..hi] — щоб кілька кривих
// лягали в одну систему координат і були порівнювані.
func drawCurveIn(screen *ebiten.Image, c *curve, x, y, w, h, lo, hi float32, col color.RGBA) {
	if c.n < 2 {
		return
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
