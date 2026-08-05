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
//	BENCH_COMBAT=0     1 = увімкнути бій (респаун стенд ставить безкінечний сам).
//	                   Вимкнений за замовчуванням: усі записані базові лінії зняті
//	                   без бою, і тихо ввімкнути його означало б їх знецінити
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
	gru    bool    // useGRU
	frames int     // memFrames: скільки слотів стеку несуть історію
	skip   int     // stackSkip: кадрів між семплами
	units  int     // скільки учнів на полі (1 перевіряє гіпотезу «рій замінює памʼять»)
	gruLR  float32 // gruLearnRate; 0 = лишити поточний
	gamma  float32 // qGamma; 0 = лишити поточний
	gskip  int     // [RNN] важіль BPTT: раз на скільки кадрів GRU думає; 0 = 1. qClip масштабується автоматично
	sight  float32 // sightRange; 0 = лишити поточний
	indep  bool    // true = sharedBrain=false (у кожного юніта СВОЯ мережа)
	wander float32 // WanderStrength; 0 = лишити конфігове (у учнів воно теж 0)
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
	// [БІЙ] Вимкнений ЗА ЗАМОВЧУВАННЯМ навмисно: усі вже записані базові лінії
	// (пул «без памʼяті» на 72 прогони, свіпи горизонту й зору) зняті без бою, і
	// увімкнути його тихо означало б знецінити їх — нові числа перестали б із ними
	// порівнюватись. Вмикати свідомо, для тих замірів, де бій і є предметом.
	combat := benchEnvInt("BENCH_COMBAT", 0) != 0

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
	if os.Getenv("BENCH_SET") == "flock" {
		// [ЧОМУ РІЙ КРАЩИЙ] Перевірка пояснення, а не самого факту.
		//
		// Факт: вісім юнітів дали 59.8% проти 53.0% в одинака. Пояснення, яке я дав
		// («агент, що загубив ціль, тримається сусідів»), виявилось ВИГАДАНИМ:
		// AlignmentRate і CohesionRate в учнів = 0, триматись нікого.
		//
		// Найімовірніша справжня причина — СПІЛЬНИЙ БУФЕР ДОСВІДУ. При восьми юнітах
		// у replay за кадр лягає вісім переходів із восьми різних місць карти.
		// Кількість кроків навчання та сама (train раз на кадр на кожну УНІКАЛЬНУ
		// мережу), але дані значно менш корельовані — а декорельованість це те,
		// заради чого experience replay і придумали.
		//
		// Контроль: вісім тіл на карті, але в кожного СВОЯ мережа зі своїм буфером.
		// Якщо перевага зникне — справа в даних, а не в кількості тіл.
		//
		// Четвертий рядок перевіряє, чи змінило щось прибирання блукання.
		cfgs = []benchCfg{
			{name: "×1  одинак", frames: 1, skip: 10, units: 1},
			{name: "×8  спільний мозок", frames: 1, skip: 10, units: 8},
			{name: "×8  СВОЯ мережа в кожного", frames: 1, skip: 10, units: 8, indep: true},
			{name: "×8  спільний + старе блукання", frames: 1, skip: 10, units: 8, wander: 0.1},
		}
	}
	if os.Getenv("BENCH_SET") == "sight" {
		// [ВИДИМІСТЬ × ПАМʼЯТЬ] Передбачення, а не просто ще один свіп.
		//
		// Сліпий кадр у нас ПОРОЖНІЙ (слоти [0..4] і [13] = 0), тож памʼять несе
		// інформацію лише коли запамʼятаний кадр був зрячим. При видимості ~10%
		// це один кадр з десяти — ось і вся стеля +3 пункти.
		//
		// ЯКЩО пояснення правильне, то з ростом sightRange розрив між «без памʼяті»
		// і «з памʼяттю» мусить РОСТИ: стане більше кадрів, у яких є що памʼятати.
		// Якщо розрив не зміниться — пояснення хибне.
		//
		// Абсолютні значення тут порівнювати не можна (зростання видимості саме по
		// собі полегшує задачу). Дивитись треба на РОЗРИВ у кожній парі.
		cfgs = []benchCfg{
			{name: "1/–    зір 260", frames: 1, skip: 10, units: 1, sight: 260},
			{name: "2/180  зір 260", frames: 2, skip: 180, units: 1, sight: 260},
			{name: "1/–    зір 400", frames: 1, skip: 10, units: 1, sight: 400},
			{name: "2/180  зір 400", frames: 2, skip: 180, units: 1, sight: 400},
			{name: "1/–    зір 600", frames: 1, skip: 10, units: 1, sight: 600},
			{name: "2/180  зір 600", frames: 2, skip: 180, units: 1, sight: 600},
		}
	}
	if os.Getenv("BENCH_SET") == "gamma" {
		// [ГОРИЗОНТ] Памʼять і планування — ПАРА, а не дві незалежні речі.
		//
		// Найкорисніше вікно памʼяті виявилось 1.5с, а горизонт цінності при
		// γ=0.95 і 120 TPS — 1/(1−γ) = 20 кадрів = 0.17с. Агент бачить півтори
		// секунди минулого, але не може оцінити переслідування довше за одну шосту
		// секунди. Гіпотеза: памʼять «не працює» саме тому, і при довшому горизонті
		// її внесок мусить вирости.
		//
		// Сітка 2×2 (без памʼяті / з памʼяттю) × (0.17с / 0.83с) плюс 1.67с зверху:
		// саме перетин відповідає на питання, а не окремі рядки.
		cfgs = []benchCfg{
			{name: "1/–    γ0.95  (0.17с)", frames: 1, skip: 10, units: 1, gamma: 0.95},
			{name: "2/180  γ0.95  (0.17с)", frames: 2, skip: 180, units: 1, gamma: 0.95},
			{name: "1/–    γ0.99  (0.83с)", frames: 1, skip: 10, units: 1, gamma: 0.99},
			{name: "2/180  γ0.99  (0.83с)", frames: 2, skip: 180, units: 1, gamma: 0.99},
			{name: "2/180  γ0.995 (1.67с)", frames: 2, skip: 180, units: 1, gamma: 0.995},
		}
	}
	if os.Getenv("BENCH_SET") == "lever" {
		// [ВАЖІЛЬ BPTT] Головна перевірка: чи справа була в тому, що градієнт не
		// дістає далі 8 кадрів.
		//
		// seqLen=8 при кроці 1 це 0.067с при 120 TPS, а задачі потрібно ~1.5с. З
		// кроком N ті самі 8 розгорнутих кроків накривають 8N кадрів:
		//   крок 5  → 40 кадрів  = 0.33с
		//   крок 20 → 160 кадрів = 1.33с   ← приблизно потрібний горизонт
		//   крок 40 → 320 кадрів = 2.67с
		//
		// Два значення, а не одне, навмисно: прорідження КОШТУЄ реактивності (між
		// рішеннями дія повторюється, тобто N кадрів латентності). Без проміжної
		// точки неможливо відрізнити «важіль допоміг» від «латентність зашкодила».
		// ПОРЯДОК ВАЖЛИВИЙ: найпотрібніші конфіги першими. Прогін триває понад
		// годину, і якщо його переб'ють, часткові результати мусять лишитись
		// придатними — а для цього ядро порівняння має бути вже пораховане.
		cfgs = []benchCfg{
			{name: "стек 2/180  ×1 (еталон)", frames: 2, skip: 180, units: 1},
			{name: "GRU крок 1   ×1 (як було)", gru: true, units: 1, gskip: 1},
			{name: "GRU крок 20  ×1 (1.33с)", gru: true, units: 1, gskip: 20},
			{name: "GRU крок 5   ×1 (0.33с)", gru: true, units: 1, gskip: 5},
			// Останнім навмисно: при кроці 40 відрізки набираються так повільно
			// (32×12×40 = 15360 кадрів лише щоб ПОЧАТИ вчитись при розігріві 30000),
			// що його провал буде неоднозначним — латентність це чи голод по даних.
			{name: "GRU крок 40  ×1 (2.67с)", gru: true, units: 1, gskip: 40},
		}
	}
	if os.Getenv("BENCH_SET") == "gru" {
		// [GRU НА ОДИНАКУ] Дірка, яку лишили попередні заміри: GRU перевірявся
		// ЛИШЕ при восьми юнітах, де рій сам вирішує задачу й памʼять не потрібна
		// нікому. Там він дав 54.8% проти 59.8% у конфігу без памʼяті — але з
		// Q ≈ 0, тобто не навчившись. Судити з цього про рекурентність не можна.
		//
		// Два темпи навчання розділяють дві причини: якщо 0.005 підтягне GRU до
		// стека — винне було наше гальмування; якщо ні — рекурентний шлях у цій
		// задачі справді слабший за стек.
		cfgs = []benchCfg{
			{name: "стек 1/–   ×1 (база)", frames: 1, skip: 10, units: 1},
			{name: "стек 2/180 ×1 (кращий)", frames: 2, skip: 180, units: 1},
			{name: "GRU lr 0.002 ×1", gru: true, frames: 4, skip: 10, units: 1, gruLR: 0.002},
			{name: "GRU lr 0.005 ×1", gru: true, frames: 4, skip: 10, units: 1, gruLR: 0.005},
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
			l, f := runBenchTrial(c, warmup, measure, moving, combat)
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
func runBenchTrial(c benchCfg, warmup, measure int, moving, combat bool) (live, frozen benchOut) {
	savedRoster, savedFrozen := unitRoster, frozenPolicy
	savedLR, savedGamma, savedClip := gruLearnRate, qGamma, qClip
	savedSight := sightRange
	savedShared := sharedBrain
	defer func() {
		sharedBrain = savedShared
		unitRoster, frozenPolicy = savedRoster, savedFrozen
		gruLearnRate, qGamma, qClip = savedLR, savedGamma, savedClip
		sightRange = savedSight
	}()

	frozenPolicy = false
	if c.gruLR > 0 {
		gruLearnRate = c.gruLR
	}
	if c.sight > 0 {
		sightRange = c.sight
	}
	if c.gamma > 0 {
		// Стеля цінності МУСИТЬ рости разом із горизонтом: рівноважна Q ≈ r/(1−γ).
		// Інакше довший горизонт уріжеться в кліп, і замір показав би «різниці немає»
		// з причини, яку ми самі й створили. Ця помилка вже раз коштувала нам вбивці
		// (природна Q≈14.8 при стелі 10).
		qClip = savedClip * (1 - savedGamma) / (1 - c.gamma)
		qGamma = c.gamma
	}

	sharedBrain = !c.indep

	// [КОНТРАКТ ПАМʼЯТІ] Тепер задається через КОНФІГ, а не через глобалі: памʼять
	// стала властивістю мережі. Стенд від цього тільки чистіший — конфіг прогону
	// описується в одному місці й не тече в глобальний стан процесу.
	learner := ConfigLearner
	learner.Count = c.units
	learner.WeightsFile = "" // ефемерні ваги: стенд не читає й не пише файли на диск
	// [РЕСПАУН] Стенд ЗАВЖДИ ставить безкінечний респаун, хоч би що стояло в конфізі
	// типу: стала популяція означає, що знаменник метрик не пливе. Без цього прогін,
	// у якому пощастило вижити, не порівнювався б із прогоном, де юнітів вибили.
	learner.Respawns = -1
	learner.Memory = MemoryStack
	if c.gru {
		learner.Memory = MemoryGRU
	}
	learner.MemFrames = c.frames
	learner.StackSkip = c.skip
	learner.GruSkip = c.gskip
	if c.wander > 0 {
		learner.WanderStrength = c.wander
	}
	unitRoster = []UnitConfig{learner}

	g := newBenchGame()
	drive := benchDriver(moving, combat)

	for i := 0; i < warmup; i++ {
		g.tickHeadless(drive, combat)
	}
	g.metrics.resetCounters()
	for i := 0; i < measure; i++ {
		g.tickHeadless(drive, combat)
	}
	live = readBench(&g.metrics)

	frozenPolicy = true
	g.metrics.resetCounters()
	for i := 0; i < measure; i++ {
		g.tickHeadless(drive, combat)
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
// БІЙ — ЗА ПРАПОРЦЕМ. Спершу його тут не було зовсім: удари на швидкості вибивали
// учнів, а респауну не існувало, і на калібруванні це з'їло цілий прогін (конфіг з
// одним юнітом втратив його на розігріві й видав рівні нулі).
//
// Тепер респаун є, і бій можна вмикати — але лише свідомо. Причина в порівнянності:
// усі записані базові лінії зняті БЕЗ бою, тож тихо його ввімкнути означало б їх
// знецінити. Стенд при цьому завжди ставить безкінечний респаун, щоб населення не
// пливло.
//
// checkCollisions не викликаємо й у бойовому режимі: playerMaxHP такий, що гравець не
// гине, а смерть перезапустила б рівень посеред вікна заміру.
func (g *Game) tickHeadless(drive func(*Game), combat bool) {
	g.tick++
	drive(g)
	g.updatePlayer()
	g.updateFlowFields()
	g.updateBoidMap()
	g.calcAcceleration()
	g.trainBrains()
	g.metrics.collect(g)
	g.updateUnits()
	if combat {
		g.resolveImpacts()
		for i := range g.units {
			g.pushOffPlayer(&g.units[i])
		}
		g.handleDeadUnits()
	}
}

// benchDriver — скриптований гравець. Не намагається бути розумним: тримає
// випадковий напрямок benchTurnEvery тіків, тоді бере новий. Об стіни не думає —
// updatePlayer сам зупиняє відповідну вісь, і виходить ковзання вздовж стін,
// схоже на живу гру. Головне, що це РУХ: рій мусить шукати ціль, а не висіти на ній.
func benchDriver(moving, combat bool) func(*Game) {
	// [БІЙ] Скриптований гравець ще й АТАКУЄ — інакше агенти з бойовою нагородою
	// (стражник, у якого вона єдина) не отримали б жодної нагородної події, і замір
	// був би про ніщо. Ритм той самий, що дозволяє гра: раз на attackCooldownMax.
	attackTick := 0
	melee := func(g *Game) {
		if !combat {
			return
		}
		attackTick++
		if attackTick >= attackCooldownMax {
			attackTick = 0
			g.applyPlayerMelee()
		}
	}

	if !moving {
		return func(g *Game) { melee(g) }
	}
	var dx, dy float32
	left := 0
	return func(g *Game) {
		melee(g)
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
