package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// ==========================================================================
// БЕЗГОЛОВИЙ СТЕНД ЗАМІРІВ ПАМʼЯТІ.
//
// Навіщо: ручні прогони в грі коштували ~9 хвилин нерухомого стояння за точку,
// і виявилось, що заморожена політика БІСТАБІЛЬНА — один і той самий конфіг дав
// 47.0% і 0.2% часу наосліп. Тобто кожен прогін — це підкидання монетки, і щоб
// побачити різницю між конфігами, потрібні десятки прогонів, а не два.
//
// Тут той самий цикл гри крутиться без графіки й без людини: гравець їздить за
// скриптом, конфіг задається змінними, прогонів — скільки скажемо.
//
// Запуск:
//
//	BOIDS_BENCH=1 go test -run TestMemoryBench -v -timeout 60m
//
// Параметри через середовище (усі необовʼязкові):
//
//	BENCH_SEEDS=10     скільки прогонів на конфіг (незалежних, див. runBenchTrial)
//	BENCH_SET=sweep    набір конфігів: main = архітектури, sweep = горизонт памʼяті
//	BENCH_WARMUP=20000 тіків навчання перед заміром
//	BENCH_MEASURE=6000 тіків у кожному вікні заміру
//	BENCH_MOVING=1     1 = гравець рухається, 0 = стоїть
//
// ЧОМУ ГРАВЕЦЬ РУХАЄТЬСЯ ЗА ЗАМОВЧУВАННЯМ. Ручний протокол вимагав стояти
// нерухомо — і це виявилось найгіршим можливим вибором: рій злипався на цілі,
// сліпих кадрів майже не лишалось, а ті, що були, — вироджені (мікрозатемнення
// в товкотнечі впритул). Памʼять у такому режимі просто не потрібна. Коли ж
// гравець їздить, рій наосліп ~90% часу — саме та задача, заради якої памʼять
// узагалі вводили.
// ==========================================================================

// benchCfg — одна конфігурація памʼяті для порівняння.
type benchCfg struct {
	name   string
	gru    bool // useGRU
	frames int  // memFrames: скільки слотів стеку несуть історію
	skip   int  // stackSkip: кадрів між семплами
	units  int  // скільки учнів на полі (1 перевіряє гіпотезу «рій замінює памʼять»)
}

// benchOut — те, що знімаємо з одного вікна заміру.
type benchOut struct {
	blind    float32 // % часу, коли агент НЕ бачив ціль — головна метрика
	chase    float32 // % сліпих кадрів із прогресом до цілі (умовна метрика)
	perAgent float32 // те саме, але усереднене по агентах
}

// benchTurnEvery — через скільки тіків скриптований гравець змінює напрямок.
// ~0.4 с при 120 TPS: досить довго, щоб реально переміщатись, і досить часто,
// щоб не застрягати в куті на весь прогін.
const benchTurnEvery = 45

func TestMemoryBench(t *testing.T) {
	if os.Getenv("BOIDS_BENCH") == "" {
		t.Skip("довгий стенд; запуск: BOIDS_BENCH=1 go test -run TestMemoryBench -v -timeout 60m")
	}
	seeds := benchEnvInt("BENCH_SEEDS", 10)
	warmup := benchEnvInt("BENCH_WARMUP", 20000)
	measure := benchEnvInt("BENCH_MEASURE", 6000)
	moving := benchEnvInt("BENCH_MOVING", 1) != 0

	// BENCH_SET=main — порівняння архітектур; sweep — горизонт памʼяті одинака.
	cfgs := []benchCfg{
		{name: "стек 1/–   ×8", frames: 1, skip: 10, units: 8},
		{name: "стек 4/10  ×8", frames: 4, skip: 10, units: 8},
		{name: "стек 4/60  ×8", frames: 4, skip: 60, units: 8},
		{name: "GRU        ×8", gru: true, frames: 4, skip: 10, units: 8},
		// Один юніт: сусідів немає, підказати нікому. Якщо памʼять і тут не дає
		// різниці — вона не потрібна задачі; якщо дає — її ховало згуртування.
		{name: "стек 1/–   ×1", frames: 1, skip: 10, units: 1},
		{name: "стек 4/10  ×1", frames: 4, skip: 10, units: 1},
	}
	if os.Getenv("BENCH_SET") == "sweep" {
		// [ГОРИЗОНТ] Скільки минулого цій задачі насправді потрібно.
		//
		// Вікно памʼяті ≈ (memFrames−1)×stackSkip тіків; при 120 TPS це секунди:
		//   4/10 → 0.25с   4/30 → 0.75с   4/60 → 1.5с   4/120 → 3.0с
		//
		// Останній рядок — КОНТРОЛЬ НА ЩІЛЬНІСТЬ: 2/180 має те саме вікно 1.5с, що
		// й 4/60, але лише ОДИН історичний семпл замість трьох. Якщо 4/60 і 2/180
		// зійдуться — вирішує горизонт; якщо 4/60 виграє — вирішує щільність. Без
		// цього рядка ці дві причини нерозрізненні, і я вже двічі сплутав їх.
		cfgs = []benchCfg{
			{name: "1/–    (без памʼяті)", frames: 1, skip: 10, units: 1},
			{name: "4/10   (0.25с)", frames: 4, skip: 10, units: 1},
			{name: "4/30   (0.75с)", frames: 4, skip: 30, units: 1},
			{name: "4/60   (1.5с)", frames: 4, skip: 60, units: 1},
			{name: "4/120  (3.0с)", frames: 4, skip: 120, units: 1},
			{name: "2/180  (1.5с, рідко)", frames: 2, skip: 180, units: 1},
		}
	}

	only := os.Getenv("BENCH_ONLY") // підрядок імені конфігу; порожньо = всі

	t.Logf("прогонів на конфіг: %d, розігрів %d тіків, вікно %d тіків, гравець рухається: %v",
		seeds, warmup, measure, moving)

	chaseBy := map[string][]float32{}
	var order []string
	for _, c := range cfgs {
		if only != "" && !strings.Contains(c.name, only) {
			continue
		}
		live := make([]float32, 0, seeds)
		frozen := make([]float32, 0, seeds)
		chase := make([]float32, 0, seeds)
		for s := 0; s < seeds; s++ {
			l, f := runBenchTrial(c, warmup, measure, moving)
			live = append(live, l.blind)
			frozen = append(frozen, f.blind)
			chase = append(chase, l.chase)
		}
		chaseBy[c.name] = chase
		order = append(order, c.name)
		t.Logf("%s | наосліп живцем %s | наосліп заморожено %s | chase живцем %s",
			c.name, benchStats(live), benchStats(frozen), benchStats(chase))
		t.Logf("%s | chase по прогонах: %s", c.name, benchList(chase))
	}

	// Попарне порівняння chase. Медіани й межі оманливі, коли вибірки
	// перекриваються: на калібруванні двох прогонів один конфіг дав напрочуд
	// вузькі [63.5..64.7] і виглядав переможцем, а на десяти зрівнявся з рештою.
	// benchDominance рахує частку ПАР прогонів, де A кращий за B — це прямо
	// відповідає на «наскільки надійно A виграє», і 50% означає «ніяк».
	t.Log("— попарно, частка пар прогонів, де перший конфіг має вищий chase —")
	for i := 0; i < len(order); i++ {
		for j := i + 1; j < len(order); j++ {
			a, b := order[i], order[j]
			t.Logf("  %s проти %s: %.0f%%", a, b, 100*benchDominance(chaseBy[a], chaseBy[b]))
		}
	}
}

// benchDominance — частка пар (a, b), де a > b (нічия = пів очка). Це
// common-language effect size: 0.5 = конфіги нерозрізненні, 1.0 = A виграє завжди.
func benchDominance(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0.5
	}
	var win float64
	for _, x := range a {
		for _, y := range b {
			switch {
			case x > y:
				win++
			case x == y:
				win += 0.5
			}
		}
	}
	return win / float64(len(a)*len(b))
}

// benchList — усі значення прогонів. Зведені числа приховують форму розподілу,
// а вона тут важлива: заморожені прогони бімодальні, і це видно лише зі списку.
func benchList(v []float32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("%.1f", x)
	}
	return strings.Join(parts, " ")
}

// runBenchTrial — один повний прогін: розігрів → вікно «живцем» → вікно заморожене.
// Міряємо ОБИДВА режими в одному прогоні: живцем показник відтворюваний, але
// слабо розрізняє конфіги; заморожений розрізняє сильно, але бістабільний.
// ПРОГОНИ НЕЗАЛЕЖНІ, А НЕ ПАРНІ. Тут стояв rand.Seed(seed), і я був певен, що
// однакові сіди дають однакові стартові умови для різних конфігів — тобто що
// порівнювати можна попарно. Це виявилось хибним: з Go 1.24 math/rand.Seed —
// ПУСТИШКА за замовчуванням (детермінізм лише з GODEBUG=randseednop=0). Три
// запуски однієї програми дали три різні числа.
//
// Наслідок для статистики: попарні тести недійсні, і мій висновок «+3.2 пункти,
// p≈0.003» був отриманий паруванням непарованих даних. Порівнювати можна лише
// незалежними методами — benchDominance і Манна-Вітні, чим і користуємось.
//
// Виклик прибрано, а не «полагоджено»: для оцінки РОЗПОДІЛУ незалежні вибірки —
// саме те, що потрібно. Якщо колись знадобиться відтворити конкретний прогін для
// відладки, запускай із GODEBUG=randseednop=0 і поверни сіди.
func runBenchTrial(c benchCfg, warmup, measure int, moving bool) (live, frozen benchOut) {
	savedRoster, savedGRU := unitRoster, useGRU
	savedSkip, savedFrames, savedFrozen := stackSkip, memFrames, frozenPolicy
	defer func() {
		unitRoster, useGRU = savedRoster, savedGRU
		stackSkip, memFrames, frozenPolicy = savedSkip, savedFrames, savedFrozen
	}()

	useGRU, stackSkip, memFrames, frozenPolicy = c.gru, c.skip, c.frames, false

	learner := ConfigLearner
	learner.Count = c.units
	learner.WeightsFile = "" // ефемерні ваги: стенд не читає й не пише файли на диск
	unitRoster = []UnitConfig{learner}

	g := newBenchGame()
	drive := benchDriver(moving)

	for i := 0; i < warmup; i++ {
		g.tickHeadless(drive)
	}
	g.metrics.resetCounters()
	for i := 0; i < measure; i++ {
		g.tickHeadless(drive)
	}
	live = readBench(&g.metrics)

	frozenPolicy = true
	g.metrics.resetCounters()
	for i := 0; i < measure; i++ {
		g.tickHeadless(drive)
	}
	frozen = readBench(&g.metrics)
	return live, frozen
}

// newBenchGame — те саме, що збирає main(), але без звуку, ворсу гравця і т.д.
func newBenchGame() *Game {
	g := &Game{
		difficulty: 1.0,
		player: Pixel{
			X: playerSpawn.X, Y: playerSpawn.Y,
			HP: playerMaxHP, MaxHP: playerMaxHP,
			Faction: factionPlayer,
		},
		units: newUnits(),
	}
	g.player.resetFur()
	return g
}

// tickHeadless — один тік логіки БЕЗ ebiten.
//
// Повторює Game.Update, окрім двох речей, що читають клавіатуру: handlePlayerInput
// (замінений на drive) і playerAttack (у замірі гравець не бʼється).
//
// БОЮ ТУТ НЕМАЄ НАВМИСНО (немає resolveImpacts/removeDeadUnits/checkCollisions).
// Гравець рухається на 5 px/кадр, тож удари на швидкості вибивають учнів, а
// респауну в нас поки немає. На калібруванні це вже з'їло цілий прогін: конфіг
// з одним юнітом втратив його ще на розігріві й видав рівні нулі. До памʼяті
// бій стосунку не має, тому просто не запускаємо його — так кількість агентів
// стала сталою, і знаменник метрики більше не залежить від везіння.
func (g *Game) tickHeadless(drive func(*Game)) {
	g.tick++
	drive(g)
	g.updatePlayer()
	g.updateFlowFields()
	g.updateBoidMap()
	g.calcAcceleration()
	g.trainBrains()
	g.metrics.collect(g)
	g.updateUnits()
}

// benchDriver — скриптований гравець. Не намагається бути розумним: тримає
// випадковий напрямок benchTurnEvery тіків, тоді бере новий. Об стіни не думає —
// updatePlayer сам зупиняє відповідну вісь, і виходить ковзання вздовж стін,
// схоже на живу гру. Головне, що це РУХ: рій мусить шукати ціль, а не висіти на ній.
func benchDriver(moving bool) func(*Game) {
	if !moving {
		return func(*Game) {}
	}
	var dx, dy float32
	left := 0
	return func(g *Game) {
		if left <= 0 {
			a := rand.Float64() * 2 * math.Pi
			dx, dy = float32(math.Cos(a)), float32(math.Sin(a))
			left = benchTurnEvery
		}
		left--
		g.player.VelX += dx * playerAccel
		g.player.VelY += dy * playerAccel
	}
}

// readBench знімає ті самі три числа, що показує панель метрик у грі.
func readBench(m *Metrics) benchOut {
	var o benchOut
	if m.unitFrames > 0 {
		o.blind = 100 * float32(m.blindN) / float32(m.unitFrames)
	}
	if m.blindN > 0 {
		o.chase = 100 * float32(m.blindClosed) / float32(m.blindN)
	}
	if pa, _, _, k := m.blindPerAgent(); k > 0 {
		o.perAgent = 100 * pa
	}
	return o
}

// benchStats — медіана й межі. Медіана, а НЕ середнє: заморожені прогони
// бістабільні (0.2% або 47%), і середнє між двома купками не описує жодну з них.
func benchStats(v []float32) string {
	if len(v) == 0 {
		return "—"
	}
	s := slices.Clone(v)
	slices.Sort(s)
	med := s[len(s)/2]
	if len(s)%2 == 0 {
		med = (s[len(s)/2-1] + s[len(s)/2]) / 2
	}
	return fmt.Sprintf("мед %5.1f%%  [%5.1f .. %5.1f]", med, s[0], s[len(s)-1])
}

func benchEnvInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
