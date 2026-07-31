package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sync"
)

// ==========================================================================
// НЕЙРОН-УЧЕНЬ v3: Q-LEARNING (REINFORCEMENT LEARNING) + WHISKERS.
//
// Що змінилось проти v2 (supervised MLP):
//   v2: вчитель казав ТОЧНУ відповідь (interceptDir) → мережа копіювала формулу.
//   v3: ВЧИТЕЛЯ НЕМАЄ. Агент діє і отримує лише reward (нагороду/штраф).
//       Стратегію він мусить відкрити САМ.
//
//   supervised:  "правильний рух = (0.83, 0.55)"      ← ми знаємо ціль
//   RL:          "ти зробив дію 3 → отримав +0.1"     ← знаємо лише оцінку
//
// Чому RL потрібен для стін: для обходу перешкод немає простої формули-вчителя.
// Зате reward тривіальний: врізався → -, наблизився → +. Агент сам зрозуміє,
// що в стіни врізатись погано, і навчиться їх обходити.
//
// [SHARED BRAIN] Поділ на Net + Brain:
//   Net   — сама мережа (ваги + target + буфер досвіду). Може бути СПІЛЬНОЮ.
//   Brain — «голова» одного ворога: указник на Net + ОСОБИСТА пам'ять агента.
//   Режим sharedBrain (main.go): усі учні ділять один Net → «вулик-розум».
//
// --------------------------------------------------------------------------
// ДВА ШЛЯХИ ПАМʼЯТІ (обираються прапорцем useGRU у рантаймі, див. Step/train):
//
//	brain_stack.go — FRAME-STACKING: вхід = стек stackFrames кадрів (56 чисел),
//	                 вікно історії задане НАМИ. Історично перший підхід; лишається
//	                 як baseline для порівняння і як fallback.
//	brain_gru.go   — GRU: вхід = ОДИН кадр (14), «минуле» живе в прихованому стані
//	                 h, і мережа САМА вчиться, що тримати й як довго. Дефолт.
//
// Тут, у brain.go, — лише СПІЛЬНЕ ядро обох: константи, Net/Brain, активації,
// ε-greedy, вуса й зір (whiskers, POMDP), anti-stuck, збереження ваг.
//
// Обидва шляхи потрібні для POMDP: агент бачить гравця лише поблизу й по прямій
// (localSight), тож без памʼяті він амнезик — забуває ціль, щойно вона за рогом.
// ==========================================================================

const (
	baseInputs  = 14 // ОДИН кадр стану: 5 базових + 8 whiskers + 1 «гравця видно»
	stackFrames = 4  // [ПАМ'ЯТЬ] скільки кадрів склеюємо на вхід (1 = без пам'яті)
	stackSkip   = 60 // кадрів між семплами історії → вікно пам'яті ≈ (stackFrames-1)*stackSkip

	brainInputs = baseInputs * stackFrames // повний вхід мережі (стек кадрів)

	// [ГЛИБИНА] Два прихованих шари — «зігнути, потім зігнути ще раз» (оріґамі).
	// Глибина ефективніше вичленяє складні залежності, ніж один ширший шар.
	brainHidden1 = 32 // 1-й прихований шар (згин простору входів)
	brainHidden2 = 16 // 2-й прихований шар (згин поверх згину)

	brainActions  = 8 // 8 напрямків руху (= кількість виходів Q)
	brainWhiskers = 8 // промені-сенсори стін

	// [RNN/GRU] Розмір рекурентного прихованого стану h (памʼять, яку мережа несе
	// між кадрами). Вмикається прапорцем useGRU: тоді вхід — ОДИН кадр (baseInputs),
	// а «минуле» живе в h, а не в стеку кадрів. Ваги GRU співіснують у Net поряд зі
	// стек-вагами (обираємо шлях у рантаймі) → перемкнути назад = нуль ризику.
	gruHidden = 32

	// [RNN крок 2] Довжина відрізка траєкторії для навчання крізь час (BPTT).
	// Рекурентна мережа вчиться на ПОСЛІДОВНОСТЯХ, а не окремих кадрах: щоб
	// «розгорнути» памʼять, треба seqLen поспіль кадрів ОДНОГО агента.
	seqLen        = 8                  // кадрів у НАВЧАЛЬНОМУ вікні (на них рахуємо loss/BPTT)
	seqBurnIn     = 4                  // кадрів ПРОГРІВУ перед вікном (лише forward, щоб h став реальним)
	seqTotal      = seqBurnIn + seqLen // усього кадрів у відрізку (12)
	seqReplaySize = 512                // місткість буфера відрізків
	seqBatch      = 8                  // відрізків на кадр у навчанні
	seqMinReplay  = 32                 // не вчимось, поки буфер не набрав стільки відрізків
	gruLearnRate  = 0.002              // [RNN] ОКРЕМА (нижча за стек) швидкість: рекурентне
	//                                    навчання вередливіше → менший крок = менше дрейфу Q
	gruGradClip = 0.5 // [RNN] кліп градієнта по часу — тугіший = спокійніший BPTT

	qLearnRate = 0.005 // швидкість навчання (RL шумніший за supervised → помірно)
	qGamma     = 0.95  // discount: наскільки цінувати майбутні нагороди (0..1)
	qClip      = 10.0  // стеля TD-цілі (захист від розбіжності)

	// [3] ε-greedy: частка ВИПАДКОВИХ дій (exploration). За замовчуванням ПОСТІЙНА
	// (qEpsilonConst). Якщо ввімкнути прапорець epsilonDecayEnabled (main.go) — ε
	// лінійно спадає max→min за qEpsilonDecay кроків (свіжий мозок досліджує →
	// з часом мисливець), але не нижче floor (qEpsilonMin).
	qEpsilonConst = 0.01  // постійна ε (коли спад ВИМКНЕНО) — поточний режим
	qEpsilonMax   = 0.30  // старт автоспаду (свіжий мозок)
	qEpsilonMin   = 0.03  // floor автоспаду — мінімум назавжди
	qEpsilonDecay = 25000 // кроків спаду max→min (~3.5 хв @120fps)

	// [DQN: стабілізатори нейро-Q-learning]
	qTargetSync = 1000 // кожні N оновлень копіюємо ваги в target-мережу
	qReplaySize = 4096 // розмір буфера досвіду (ring buffer)
	qBatch      = 16   // скільки випадкових переходів вчимо щокадру
	qMinReplay  = 200  // не вчимось, поки буфер не набрав стільки переходів

	brainMaxWeight = 5.0 // стеля ваг
	brainForce     = 0.3 // масштаб дії у прискорення

	whiskerRange = 110.0 // далекість «вусів» у px (~5 тайлів)
	whiskerStep  = 4.0   // крок променя при пошуку стіни

	sightRange = 260.0 // [POMDP] радіус видимості гравця (px) у режимі localSight

	// Масштаб reward підібраний так, щоб «хороший» кадр давав сигнал ~0.5,
	// а удар об стіну — помітний штраф. Замалий reward = TD-сигнал тоне в шумі.
	rewardCloserScale = 0.5  // нагорода за наближення до гравця (на px/кадр)
	rewardWallHit     = -1.0 // штраф за удар об стіну (по факту зіткнення)
	rewardNearWall    = -0.3 // [1] штраф за рух У БІК близької стіни (плавний градієнт обходу)

	// [2] anti-stuck «фрустрація»: коли агент застряг (б'ється в стіну АБО
	// притиснутий до неї без прогресу) — на frustrationFrames кадрів форсуємо
	// НАПРАВЛЕНИЙ вихід у найвідкритіший бік (escapeAction), а не випадковий смик.
	stuckHitInc       = 4    // +N за кадр з ударом об стіну
	stuckNoProgInc    = 1    // +N за кадр «біля стіни без наближення до гравця»
	stuckDecay        = 1    // -N коли все гаразд (поступове забування)
	stuckLimit        = 24   // поріг лічильника → вмикається фрустрація
	frustrationFrames = 24   // скільки кадрів напрямленого виходу
	stuckNearWall     = 0.5  // whisker ≥ цього = «біля стіни»
	stuckProgressEps  = 0.05 // наближення менше за це = «нема прогресу»
)

// sqrt2inv = 1/√2 — для діагональних напрямків (щоб були одиничної довжини).
const sqrt2inv = 0.70710678

// dirs8 — 8 напрямків руху за годинниковою (екранні координати: y росте ВНИЗ).
// Whisker[i] вимірює стіну в напрямку dirs8[i]; action[i] рухає туди ж —
// тобто мережа «бачить», наскільки заблокований кожен напрямок, у який може піти.
//
//	0=N(вгору) 1=NE 2=E(вправо) 3=SE 4=S(вниз) 5=SW 6=W(вліво) 7=NW
var dirs8 = [brainActions][2]float32{
	{0, -1}, {sqrt2inv, -sqrt2inv}, {1, 0}, {sqrt2inv, sqrt2inv},
	{0, 1}, {-sqrt2inv, sqrt2inv}, {-1, 0}, {-sqrt2inv, -sqrt2inv},
}

// Net — НЕЙРОМЕРЕЖА Q-агента: ваги + target-копія + буфер досвіду.
//
// Кілька Brain можуть указувати на ОДИН Net (режим sharedBrain=true) → «вулик-
// розум»: усі ділять одну вивчену політику й спільний досвід.
//
// [GO: ВАГИ ТРЬОХ ШАРІВ]
// W1: вхід → hidden1;  W2: hidden1 → hidden2;  W3: hidden2 → вихід(Q).
type Net struct {
	W1 [brainHidden1][brainInputs]float32
	B1 [brainHidden1]float32
	W2 [brainHidden2][brainHidden1]float32
	B2 [brainHidden2]float32
	W3 [brainActions][brainHidden2]float32
	B3 [brainActions]float32

	// [DQN: TARGET NETWORK] заморожена копія для Беллман-цілі (щоб не «тікала»).
	tW1         [brainHidden1][brainInputs]float32
	tB1         [brainHidden1]float32
	tW2         [brainHidden2][brainHidden1]float32
	tB2         [brainHidden2]float32
	tW3         [brainActions][brainHidden2]float32
	tB3         [brainActions]float32
	syncCounter int

	// [DQN: EXPERIENCE REPLAY] кільцевий буфер переходів (frame-stacking шлях).
	replay     []transition
	replayHead int
	replayFull bool

	// [RNN] Кільцевий буфер ВІДРІЗКІВ для рекурентного навчання (GRU шлях).
	// Той самий mu стереже обидва буфери (пишуть з паралельної фази).
	seqReplay []sequence
	seqHead   int
	seqFull   bool

	// [GO: MUTEX] захищає СПІЛЬНИЙ буфер від одночасного запису з різних горутин
	// (remember у паралельній фазі calcAcceleration). Ваги ж безпечні без локу
	// через РОЗДІЛЕННЯ ФАЗ: forward читається паралельно, train пише однопотоково
	// (g.trainBrains) — фази не перетинаються.
	mu sync.Mutex

	// [МЕТРИКИ] акумулятори за період (скидаються в Metrics.collect). Пишуться
	// у tdUpdate (однопотоково в trainBrains) → без локу.
	mTDSum float32 // сума |TD-error| (сирого) — «здивування» мережі
	mQSum  float32 // сума max Q(s) — канарка розбіжності (росте безмежно = біда)
	mTDN   int     // кількість оновлень за період

	// [RNN/GRU] Ваги рекурентної клітини (вживаються лише коли useGRU=true).
	// GRU-клітина: вхід x(baseInputs) + попередній стан h(gruHidden) → новий h.
	//   z — update gate (скільки нового пускати в памʼять)
	//   r — reset gate (скільки старого забути перед оновленням)
	//   h~ — candidate (кандидат нового стану)
	// W* множать ВХІД, U* множать СТАН, B* — зсуви. Wq/Bq: стан h → Q(8) (лінійно).
	Wz [gruHidden][baseInputs]float32
	Uz [gruHidden][gruHidden]float32
	Bz [gruHidden]float32
	Wr [gruHidden][baseInputs]float32
	Ur [gruHidden][gruHidden]float32
	Br [gruHidden]float32
	Wh [gruHidden][baseInputs]float32
	Uh [gruHidden][gruHidden]float32
	Bh [gruHidden]float32
	Wq [brainActions][gruHidden]float32
	Bq [brainActions]float32

	// [RNN] Target-копії GRU-ваг — заморожені для Беллман-цілі (як tW1… для стеку).
	tWz [gruHidden][baseInputs]float32
	tUz [gruHidden][gruHidden]float32
	tBz [gruHidden]float32
	tWr [gruHidden][baseInputs]float32
	tUr [gruHidden][gruHidden]float32
	tBr [gruHidden]float32
	tWh [gruHidden][baseInputs]float32
	tUh [gruHidden][gruHidden]float32
	tBh [gruHidden]float32
	tWq [brainActions][gruHidden]float32
	tBq [brainActions]float32
}

// Brain — «голова» одного ворога-учня: указник на мережу + ОСОБИСТА пам'ять.
// Мережа може бути спільною; пам'ять (стан у часі, лічильники) — завжди своя,
// тож агенти діють індивідуально, але вчаться в (можливо) спільну мережу.
type Brain struct {
	net *Net

	// [RL: ПАМ'ЯТЬ МІЖ КАДРАМИ] — у кожного агента своя.
	// Reward за дію відомий лише НАСТУПНОГО кадру (коли побачимо результат руху).
	prevState  [brainInputs]float32 // попередній СТЕКНУТИЙ стан (для переходу)
	prevAction int
	prevDist   float32
	hasPrev    bool

	// [ПАМ'ЯТЬ] Історія кадрів для frame-stacking: семпли кожні stackSkip кадрів.
	// Повний вхід = [поточний кадр | frames[0] | frames[1] | ...].
	frames    [stackFrames - 1][baseInputs]float32
	frameTick int

	// [RNN/GRU] Рекурентний прихований стан цього агента (вживається при useGRU).
	// Несеться між кадрами, скидається на новий епізод/respawn. Памʼять — своя в
	// кожного агента (як frames); ваги GRU — спільні в Net (вулик лишається).
	h [gruHidden]float32

	// [RNN крок 2] Накопичувач поточного відрізка траєкторії. Коли набереться
	// seqLen завершених кроків — відрізок їде в Net.seqReplay, лічильник у 0.
	gruPrevX [baseInputs]float32 // попередній вхідний кадр (для запису кроку)
	seqX     [seqTotal][baseInputs]float32
	seqA     [seqTotal]int
	seqR     [seqTotal]float32
	seqN     int

	age int // [3] вік (к-сть Step) — для автоспаду ε

	// [2] anti-stuck: лічильник застрягання, залишок кадрів «фрустрації», прапорець сліду.
	stuckCounter int
	frustration  int
	markStuck    bool

	// [SELF-PLAY] flee=true → ЖЕРТВА: reward інвертується (далі від ворога = краще).
	// false → ХИЖАК (ближче до гравця = краще), як у ворогів.
	flee bool

	// Для візуалізації (читає Draw, пише calcAcceleration — різні фази, без гонки).
	lastWhiskers [brainWhiskers]float32
	lastAction   int
	lastReward   float32 // [МЕТРИКИ] нагорода останнього кроку (для середнього по рою)

	// [МЕТРИКИ ПАМʼЯТІ] Blind-pursuit: чи бачив агент гравця, КОЛИ обирав минулу
	// дію (visible на момент рішення). Дозволяє поміряти: коли агент СЛІПИЙ, чи
	// продовжує він скорочувати дистанцію (памʼять) чи блукає (реактивний амнезик).
	// Лічильники пише лише власна горутина агента (paralel-фаза) → без гонок;
	// collect() підсумовує їх однопотоково й скидає (як lastReward).
	prevVisible  bool
	mBlindN      int // «сліпих рішень» за період
	mBlindClosed int // ...із них скоротили дистанцію до гравця
}

// tanh — активація прихованого шару. Похідна: tanh'(z) = 1 - tanh(z)².
func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// sigmoid — активація воріт GRU, стискає в (0,1) = «скільки пропустити».
// Похідна: σ'(z) = σ(z)·(1−σ(z)) — знадобиться для BPTT (крок 3).
func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

// NewNet створює мережу з Xavier-ініціалізацією (масштаб ~1/√fan_in),
// щоб tanh не входив у насичення і градієнт не зникав.
func NewNet() *Net {
	n := &Net{}
	s1 := float32(math.Sqrt(1.0 / brainInputs))
	for j := range n.W1 {
		for i := range n.W1[j] {
			n.W1[j][i] = (rand.Float32()*2 - 1) * s1
		}
	}
	s2 := float32(math.Sqrt(1.0 / brainHidden1))
	for k := range n.W2 {
		for j := range n.W2[k] {
			n.W2[k][j] = (rand.Float32()*2 - 1) * s2
		}
	}
	s3 := float32(math.Sqrt(1.0 / brainHidden2))
	for a := range n.W3 {
		for k := range n.W3[a] {
			n.W3[a][k] = (rand.Float32()*2 - 1) * s3
		}
	}
	n.initGRU()    // [RNN] ініціалізуємо й рекурентні ваги (навіть якщо useGRU=false)
	n.syncTarget() // target стартує копією живих ваг
	return n
}

// NewBrain — голова агента з ВЛАСНОЮ новою мережею (незалежний режим).
func NewBrain() *Brain { return &Brain{net: NewNet()} }

// NewBrainWith — голова агента, що ДІЛИТЬ передану мережу (режим sharedBrain).
func NewBrainWith(net *Net) *Brain { return &Brain{net: net} }

// syncTarget копіює живі ваги в target-мережу.
// [GO: масиви — значимі типи] присвоєння масиву копіює його повністю.
func (n *Net) syncTarget() {
	n.tW1, n.tB1 = n.W1, n.B1
	n.tW2, n.tB2 = n.W2, n.B2
	n.tW3, n.tB3 = n.W3, n.B3
	// [RNN] і GRU-ваги
	n.tWz, n.tUz, n.tBz = n.Wz, n.Uz, n.Bz
	n.tWr, n.tUr, n.tBr = n.Wr, n.Ur, n.Br
	n.tWh, n.tUh, n.tBh = n.Wh, n.Uh, n.Bh
	n.tWq, n.tBq = n.Wq, n.Bq
}

// argmaxQ повертає індекс дії з найбільшим Q.
func argmaxQ(q [brainActions]float32) int {
	best := 0
	for k := 1; k < brainActions; k++ {
		if q[k] > q[best] {
			best = k
		}
	}
	return best
}

// maxWhisker — найбільша близькість стіни серед 8 променів (0 = чисто навкруги).
// Читає вуса ПОТОЧНОГО кадру (індекси 5..12) — вони однакові і в стеку, і в GRU.
func maxWhisker(f [baseInputs]float32) float32 {
	m := f[5]
	for i := 1; i < brainWhiskers; i++ {
		if f[5+i] > m {
			m = f[5+i]
		}
	}
	return m
}

// escapeAction — напрямок із НАЙМЕНШОЮ близькістю стіни (найвідкритіший), щоб
// гарантовано вийти з пастки, а не смикатись на місці. Серед однаково відкритих
// напрямків — рівноймовірно (reservoir), аби агенти не злипались в один бік.
func escapeAction(f [baseInputs]float32) int {
	best, bestW, ties := 0, f[5], 1
	for i := 1; i < brainWhiskers; i++ {
		w := f[5+i]
		switch {
		case w < bestW:
			best, bestW, ties = i, w, 1
		case w == bestW:
			ties++
			if rand.Intn(ties) == 0 {
				best = i
			}
		}
	}
	return best
}

// epsilon — поточна ε з лінійним автоспадом max→min за qEpsilonDecay кроків,
// але не нижче floor (qEpsilonMin). Свіжий мозок (age=0) досліджує найбільше,
// натренований (age велике / завантажений) — майже чистий мисливець.
func (b *Brain) epsilon() float32 {
	if !epsilonDecayEnabled {
		return qEpsilonConst // спад вимкнено → ε постійна, не змінюється
	}
	t := float32(b.age) / qEpsilonDecay
	if t > 1 {
		t = 1
	}
	return qEpsilonMax + (qEpsilonMin-qEpsilonMax)*t
}

// selectAction — ε-GREEDY вибір дії (читає ваги мережі).
//
// [RL: EXPLORATION vs EXPLOITATION]
// З імовірністю ε — випадкова дія (досліджуємо світ). Інакше — найкраща за Q.
func (b *Brain) selectAction(state [brainInputs]float32) int {
	if rand.Float32() < b.epsilon() {
		return rand.Intn(brainActions)
	}
	q, _, _ := b.net.forwardQ(state)
	return argmaxQ(q)
}

// selectFromQ — ε-greedy на ВЖЕ обрахованому q-векторі (шлях GRU: q дає forwardGRU,
// повторно рахувати не треба). Та сама логіка, що й selectAction.
func (b *Brain) selectFromQ(q [brainActions]float32) int {
	if rand.Float32() < b.epsilon() {
		return rand.Intn(brainActions)
	}
	return argmaxQ(q)
}

// Step — один крок агента: оцінити минулу дію й обрати нову. Викликається
// щокадру в паралельній фазі (calcAcceleration). НЕ тренує мережу — лише кладе
// досвід у буфер; навчання відбувається раз/кадр однопотоково в g.trainBrains()
// (мережа може бути спільною: тренувати її N×qBatch/кадр було б і неправильно,
// і небезпечно для гонок).
//
// [ДИСПЕТЧЕР] Тут обирається ШЛЯХ ПАМʼЯТІ: рекурентний GRU (brain_gru.go) або
// frame-stacking (brain_stack.go). Усе інше в них — незалежне.
func (b *Brain) Step(cur [baseInputs]float32, dist float32, hitWall bool) int {
	b.age++ // [3] для автоспаду ε
	if useGRU {
		return b.stepGRU(cur, dist, hitWall)
	}
	return b.stepStack(cur, dist, hitWall)
}

// train — k оновлень на випадкових вибірках із буфера (серце DQN).
//
// Викликається ОДИН раз за кадр ОДНОПОТОКОВО (g.trainBrains, поза паралельною
// фазою) → запис ваг безпечний без локу. Зі спільним мозком уся колективна
// вибірка тренує ОДНУ мережу нормальним темпом (а не N×qBatch разів за кадр).
//
// [ДИСПЕТЧЕР] Той самий поділ шляхів, що й у Step.
func (n *Net) train(k int) {
	if useGRU {
		n.trainSeq(seqBatch) // [RNN] рекурентний шлях — навчання на відрізках (BPTT)
		return
	}
	m := n.replayLen()
	if m < qMinReplay {
		return
	}
	for i := 0; i < k; i++ {
		t := n.replay[rand.Intn(m)]
		n.tdUpdate(t.s, t.a, t.r, t.s2)
	}
}

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// wallWhisker — промінь-сенсор: відстань від точки (cx,cy) до стіни в напрямку (dx,dy).
//
// [AI: RAYCAST]
// Крокуємо вздовж напрямку, поки не натрапимо на стіну (або не вийдемо за range).
// Повертаємо БЛИЗЬКІСТЬ: 1 = стіна впритул, 0 = чисто на всю довжину.
func wallWhisker(cx, cy, dx, dy float32) float32 {
	for d := float32(whiskerStep); d <= whiskerRange; d += whiskerStep {
		px := cx + dx*d
		py := cy + dy*d
		if isWallAt(int(px)/pixelSize, int(py)/pixelSize) {
			return 1 - d/whiskerRange
		}
	}
	return 0
}

// hasLineOfSight — чи є пряма видимість між точками (немає стіни на прямій).
// Крокуємо від (x1,y1) до (x2,y2) по пів-клітинки; якщо натрапили на стіну до
// цілі — видимості нема. [POMDP] так гравець «ховається» за стінами.
func hasLineOfSight(x1, y1, x2, y2 float32) bool {
	dx := x2 - x1
	dy := y2 - y1
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	steps := int(dist / (pixelSize / 2))
	if steps < 1 {
		return true
	}
	sx := dx / float32(steps)
	sy := dy / float32(steps)
	x, y := x1, y1
	for i := 0; i < steps; i++ {
		x += sx
		y += sy
		if isWallAt(int(x)/pixelSize, int(y)/pixelSize) {
			return false
		}
	}
	return true
}

// GatherInputs збирає стан (state) для Q-мережі.
//
//	[0] dx/screenWidth     напрямок до гравця X   ◄─┐
//	[1] dy/screenHeight    напрямок до гравця Y     │ обнуляються,
//	[2] dist/screenWidth   відстань до гравця       │ коли гравця НЕ видно
//	[3] pVelX/5            швидкість гравця X        │ (POMDP)
//	[4] pVelY/5            швидкість гравця Y     ◄─┘
//	[5..12] whiskers       близькість стіни у 8 напрямках  ◄── зір на перешкоди (завжди)
//	[13] visible           1 = гравця видно, 0 = ні
//
// [POMDP] Якщо localSight — гравець «видимий» лише в межах sightRange і по прямій
// видимості (промінь не перекритий стіною). Поза цим позиційні входи = 0 і
// visible = 0 → агент не знає, де гравець. Саме тут згодом порятує пам'ять.
//
// Побічно зберігає whiskers у Brain для візуалізації.
// Повертає ОДИН кадр (baseInputs); склеювання в стек робить Brain.Step.
func GatherInputs(enemy, player *Pixel) [baseInputs]float32 {
	dx := player.X - enemy.X
	dy := player.Y - enemy.Y
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if dist == 0 {
		dist = 1
	}

	cx := enemy.X + pixelSize/2
	cy := enemy.Y + pixelSize/2

	visible := true
	if localSight {
		visible = dist <= sightRange &&
			hasLineOfSight(cx, cy, player.X+pixelSize/2, player.Y+pixelSize/2)
	}

	var in [baseInputs]float32
	if visible {
		in[0] = dx / screenWidth
		in[1] = dy / screenHeight
		in[2] = dist / screenWidth
		in[3] = player.VelX / 5.0
		in[4] = player.VelY / 5.0
		in[13] = 1
	}
	// (якщо не видно — [0..4] і [13] лишаються 0)

	for i := 0; i < brainWhiskers; i++ {
		w := wallWhisker(cx, cy, dirs8[i][0], dirs8[i][1])
		in[5+i] = w
		if enemy.Brain != nil {
			enemy.Brain.lastWhiskers[i] = w
		}
	}
	return in
}

// GatherPreyInputs — стан для мозку-ЖЕРТВИ (гравця у self-play). Дзеркало
// GatherInputs: замість «куди гравець» — «звідки загроза» (напрямок до НАЙБЛИЖЧОГО
// ворога). Мережа вчиться рухатись ГЕТЬ (бо reward інвертований, flee=true).
// Повертає кадр (baseInputs) і відстань до найближчого ворога (для reward).
func GatherPreyInputs(player *Pixel, enemies []Pixel) ([baseInputs]float32, float32) {
	cx := player.X + pixelSize/2
	cy := player.Y + pixelSize/2

	// Найближчий ворог = головна загроза.
	nearest := -1
	var best float32 = 1e30
	for i := range enemies {
		dx := enemies[i].X - player.X
		dy := enemies[i].Y - player.Y
		d := dx*dx + dy*dy
		if d < best {
			best = d
			nearest = i
		}
	}

	var in [baseInputs]float32
	dist := float32(1)
	if nearest >= 0 {
		e := &enemies[nearest]
		dx := e.X - player.X
		dy := e.Y - player.Y
		dist = float32(math.Sqrt(float64(dx*dx + dy*dy)))
		if dist == 0 {
			dist = 1
		}
		in[0] = dx / screenWidth
		in[1] = dy / screenHeight
		in[2] = dist / screenWidth
		in[3] = e.VelX / 5.0
		in[4] = e.VelY / 5.0
		in[13] = 1
	}

	for i := 0; i < brainWhiskers; i++ {
		w := wallWhisker(cx, cy, dirs8[i][0], dirs8[i][1])
		in[5+i] = w
		if player.Brain != nil {
			player.Brain.lastWhiskers[i] = w
		}
	}
	return in, dist
}

// brainFile — шлях до файлу де зберігаються вивчені ваги між сесіями.
const brainFile = "brain_weights.json"

// BrainData — серіалізація ваг у JSON + розміри мережі для перевірки сумісності.
type BrainData struct {
	Inputs  int `json:"inputs"`
	Hidden1 int `json:"hidden1"`
	Hidden2 int `json:"hidden2"`
	Actions int `json:"actions"`

	W1 [brainHidden1][brainInputs]float32  `json:"w1"`
	B1 [brainHidden1]float32               `json:"b1"`
	W2 [brainHidden2][brainHidden1]float32 `json:"w2"`
	B2 [brainHidden2]float32               `json:"b2"`
	W3 [brainActions][brainHidden2]float32 `json:"w3"`
	B3 [brainActions]float32               `json:"b3"`

	// [RNN] Ваги GRU. Старі файли їх не містять (HasGRU=false) → GRU стартує з нуля
	// через initGRU. GruHidden звіряємо окремо (розмір h) при завантаженні.
	HasGRU    bool                             `json:"has_gru"`
	GruHidden int                              `json:"gru_hidden"`
	Wz        [gruHidden][baseInputs]float32   `json:"wz"`
	Uz        [gruHidden][gruHidden]float32    `json:"uz"`
	Bz        [gruHidden]float32               `json:"bz"`
	Wr        [gruHidden][baseInputs]float32   `json:"wr"`
	Ur        [gruHidden][gruHidden]float32    `json:"ur"`
	Br        [gruHidden]float32               `json:"br"`
	Wh        [gruHidden][baseInputs]float32   `json:"wh"`
	Uh        [gruHidden][gruHidden]float32    `json:"uh"`
	Bh        [gruHidden]float32               `json:"bh"`
	Wq        [brainActions][gruHidden]float32 `json:"wq"`
	Bq        [brainActions]float32            `json:"bq"`
}

// SaveBrain зберігає ваги мережі агента у JSON.
func SaveBrain(b *Brain) error { return SaveNet(b.net) }

// SaveNet зберігає ваги мережі у файл за замовчуванням (brainFile).
func SaveNet(n *Net) error { return saveNetTo(n, brainFile) }

// saveNetTo серіалізує мережу (стек + GRU ваги) у JSON за вказаним шляхом.
func saveNetTo(n *Net, path string) error {
	data := BrainData{
		Inputs: brainInputs, Hidden1: brainHidden1, Hidden2: brainHidden2, Actions: brainActions,
		W1: n.W1, B1: n.B1, W2: n.W2, B2: n.B2, W3: n.W3, B3: n.B3,
		// [RNN] і рекурентні ваги — щоб gru-рій не вчився з нуля щоразу.
		HasGRU: true, GruHidden: gruHidden,
		Wz: n.Wz, Uz: n.Uz, Bz: n.Bz,
		Wr: n.Wr, Ur: n.Ur, Br: n.Br,
		Wh: n.Wh, Uh: n.Uh, Bh: n.Bh,
		Wq: n.Wq, Bq: n.Bq,
	}
	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, bytes, 0644)
}

// LoadNet завантажує мережу з файлу за замовчуванням (brainFile).
func LoadNet() *Net { return loadNetFrom(brainFile) }

// loadNetFrom завантажує мережу з файлу. Повертає nil (→ caller створить NewNet),
// якщо файлу немає, він пошкоджений, або РОЗМІРИ стек-мережі не збігаються.
// GRU-ваги вантажимо, ЛИШЕ якщо файл їх містить і розмір h збігається; інакше —
// initGRU (стара збірка чи інший gruHidden → рекурентна памʼять з нуля, стек цілий).
func loadNetFrom(path string) *Net {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var data BrainData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil
	}
	if data.Inputs != brainInputs || data.Hidden1 != brainHidden1 ||
		data.Hidden2 != brainHidden2 || data.Actions != brainActions {
		return nil // несумісна архітектура → почнемо з нуля
	}
	n := &Net{W1: data.W1, B1: data.B1, W2: data.W2, B2: data.B2, W3: data.W3, B3: data.B3}
	if data.HasGRU && data.GruHidden == gruHidden {
		n.Wz, n.Uz, n.Bz = data.Wz, data.Uz, data.Bz
		n.Wr, n.Ur, n.Br = data.Wr, data.Ur, data.Br
		n.Wh, n.Uh, n.Bh = data.Wh, data.Uh, data.Bh
		n.Wq, n.Bq = data.Wq, data.Bq
	} else {
		n.initGRU() // немає ваг GRU у файлі / інший розмір → рекурентна памʼять з нуля
	}
	n.syncTarget()
	return n
}
