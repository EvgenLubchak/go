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
// АРХІТЕКТУРА (Q-мережа):
//   state = стек кадрів (baseInputs×stackFrames) → hidden1(32) → hidden2(16) → Q(8)
//   baseInputs(14) = 5 базових (dx,dy,dist,pVelX,pVelY) + 8 whiskers + 1 «гравця видно»
//   8 виходів = Q-значення для 8 напрямків руху. Дія = напрямок з найбільшим Q.
//
// [ПАМ'ЯТЬ] frame-stacking: мережа бачить не лише «зараз», а й недавнє минуле
// (семпли кожні stackSkip кадрів) → може ПАМ'ЯТАТИ, куди зник гравець. Це крок від
// реактивного (амнезія щокадру) агента до роботи з частковою спостережуваністю (POMDP).
//
// [SHARED BRAIN] Поділ на Net + Brain:
//   Net   — сама мережа (ваги + target + буфер досвіду). Може бути СПІЛЬНОЮ.
//   Brain — «голова» одного ворога: указник на Net + ОСОБИСТА пам'ять агента.
//   Режим sharedBrain (main.go): усі учні ділять один Net → «вулик-розум».
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
	seqLen        = 8   // кадрів у відрізку
	seqReplaySize = 512 // місткість буфера відрізків (≈ seqLen×512 кадрів)
	seqBatch      = 8   // відрізків на кадр у навчанні (кожен = seqLen кроків BPTT)
	seqMinReplay  = 32  // не вчимось, поки буфер не набрав стільки відрізків
	gruGradClip   = 1.0 // [RNN] кліп градієнта по часу — проти вибуху при BPTT

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

// transition — один крок досвіду: (стан, дія, нагорода, наступний стан).
// Це «одиниця пам'яті» для experience replay (frame-stacking шлях).
type transition struct {
	s  [brainInputs]float32
	a  int
	r  float32
	s2 [brainInputs]float32
}

// sequence — [RNN] відрізок траєкторії ОДНОГО агента: seqLen поспіль кадрів
// (вхід x, дія a, нагорода r) + xEnd (кадр ПІСЛЯ останнього кроку, для bootstrap
// Беллман-цілі). Це «одиниця пам'яті» рекурентного навчання: BPTT (крок 3)
// прожене GRU по цих кадрах від нульового стану й порахує TD на кожному.
type sequence struct {
	x    [seqLen][baseInputs]float32
	a    [seqLen]int
	r    [seqLen]float32
	xEnd [baseInputs]float32
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
	seqX     [seqLen][baseInputs]float32
	seqA     [seqLen]int
	seqR     [seqLen]float32
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

// initGRU — Xavier-ініціалізація ваг рекурентної клітини. Викликається і з NewNet,
// і з LoadNet (файл ваг GRU поки не містить → щоб не лишались нульовими й мертвими).
// Вхідні ваги масштабуємо ~1/√baseInputs, рекурентні й вихідні ~1/√gruHidden.
func (n *Net) initGRU() {
	sx := float32(math.Sqrt(1.0 / baseInputs))
	sh := float32(math.Sqrt(1.0 / gruHidden))
	rnd := func(scale float32) float32 { return (rand.Float32()*2 - 1) * scale }

	for i := 0; i < gruHidden; i++ {
		for j := 0; j < baseInputs; j++ {
			n.Wz[i][j], n.Wr[i][j], n.Wh[i][j] = rnd(sx), rnd(sx), rnd(sx)
		}
		for j := 0; j < gruHidden; j++ {
			n.Uz[i][j], n.Ur[i][j], n.Uh[i][j] = rnd(sh), rnd(sh), rnd(sh)
		}
	}
	for a := 0; a < brainActions; a++ {
		for k := 0; k < gruHidden; k++ {
			n.Wq[a][k] = rnd(sh)
		}
	}
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

// forwardQTarget — як forwardQ, але по ЗАМОРОЖЕНИХ (target) вагах.
// Використовується лише для обрахунку Беллман-цілі.
func (n *Net) forwardQTarget(state [brainInputs]float32) [brainActions]float32 {
	var h1 [brainHidden1]float32
	for j := 0; j < brainHidden1; j++ {
		z := n.tB1[j]
		for i := 0; i < brainInputs; i++ {
			z += state[i] * n.tW1[j][i]
		}
		h1[j] = tanh(z)
	}
	var h2 [brainHidden2]float32
	for k := 0; k < brainHidden2; k++ {
		z := n.tB2[k]
		for j := 0; j < brainHidden1; j++ {
			z += h1[j] * n.tW2[k][j]
		}
		h2[k] = tanh(z)
	}
	var q [brainActions]float32
	for a := 0; a < brainActions; a++ {
		z := n.tB3[a]
		for k := 0; k < brainHidden2; k++ {
			z += h2[k] * n.tW3[a][k]
		}
		q[a] = z
	}
	return q
}

// forwardQ — прямий прохід: state → hidden1 → hidden2 → Q-значення всіх 8 дій.
//
// [RL: Q(s, a) = "наскільки хороша дія a у стані s"]
// Два приховані шари з tanh (два «згини» простору), вихід ЛІНІЙНИЙ. Повертає
// також активації h1, h2 — вони потрібні для backprop. Функція ЧИСТА (лише читає
// ваги) → безпечна для паралельного виклику, поки ніхто не ПИШЕ ваги.
func (n *Net) forwardQ(state [brainInputs]float32) (q [brainActions]float32, h1 [brainHidden1]float32, h2 [brainHidden2]float32) {
	for j := 0; j < brainHidden1; j++ {
		z := n.B1[j]
		for i := 0; i < brainInputs; i++ {
			z += state[i] * n.W1[j][i]
		}
		h1[j] = tanh(z)
	}
	for k := 0; k < brainHidden2; k++ {
		z := n.B2[k]
		for j := 0; j < brainHidden1; j++ {
			z += h1[j] * n.W2[k][j]
		}
		h2[k] = tanh(z)
	}
	for a := 0; a < brainActions; a++ {
		z := n.B3[a]
		for k := 0; k < brainHidden2; k++ {
			z += h2[k] * n.W3[a][k]
		}
		q[a] = z // лінійний вихід
	}
	return
}

// forwardGRU — один крок рекурентної клітини: (вхід x, старий стан hPrev) →
// (Q-значення, НОВИЙ стан hNew). Формули стандартного GRU:
//
//	z  = σ(Wz·x + Uz·h + Bz)          update gate  — скільки нового пускати
//	r  = σ(Wr·x + Ur·h + Br)          reset  gate  — скільки старого забути
//	h~ = tanh(Wh·x + Uh·(r⊙h) + Bh)   кандидат нового стану
//	h' = (1−z)⊙h + z⊙h~               новий стан (інтерполяція старе↔кандидат)
//	q  = Wq·h' + Bq                   Q-значення (лінійно)
//
// Функція ЧИСТА (лише читає ваги) → безпечна для паралельного виклику.
func (n *Net) forwardGRU(x [baseInputs]float32, hPrev [gruHidden]float32) (q [brainActions]float32, hNew [gruHidden]float32) {
	// Прохід 1: ворота z і r (обидва — повні вектори; кандидат нижче потребує
	// ВЕСЬ r, бо reset діє поелементно на весь стан: (r⊙h)_j = r_j·h_j).
	var z, r [gruHidden]float32
	for i := 0; i < gruHidden; i++ {
		zi, ri := n.Bz[i], n.Br[i]
		for j := 0; j < baseInputs; j++ {
			zi += n.Wz[i][j] * x[j]
			ri += n.Wr[i][j] * x[j]
		}
		for j := 0; j < gruHidden; j++ {
			zi += n.Uz[i][j] * hPrev[j]
			ri += n.Ur[i][j] * hPrev[j]
		}
		z[i] = sigmoid(zi)
		r[i] = sigmoid(ri)
	}

	// Прохід 2: кандидат h~ (з reset-gated станом r⊙h) і новий стан h'.
	for i := 0; i < gruHidden; i++ {
		ci := n.Bh[i]
		for j := 0; j < baseInputs; j++ {
			ci += n.Wh[i][j] * x[j]
		}
		for j := 0; j < gruHidden; j++ {
			ci += n.Uh[i][j] * (r[j] * hPrev[j])
		}
		cand := tanh(ci)
		hNew[i] = (1-z[i])*hPrev[i] + z[i]*cand
	}

	// Вихід: Q-значення зі стану (лінійно).
	for a := 0; a < brainActions; a++ {
		s := n.Bq[a]
		for k := 0; k < gruHidden; k++ {
			s += n.Wq[a][k] * hNew[k]
		}
		q[a] = s
	}
	return
}

// forwardGRUTarget — як forwardGRU, але по ЗАМОРОЖЕНИХ (target) GRU-вагах.
// Для обрахунку Беллман-цілі при BPTT (крок 3).
func (n *Net) forwardGRUTarget(x [baseInputs]float32, hPrev [gruHidden]float32) (q [brainActions]float32, hNew [gruHidden]float32) {
	var z, r [gruHidden]float32
	for i := 0; i < gruHidden; i++ {
		zi, ri := n.tBz[i], n.tBr[i]
		for j := 0; j < baseInputs; j++ {
			zi += n.tWz[i][j] * x[j]
			ri += n.tWr[i][j] * x[j]
		}
		for j := 0; j < gruHidden; j++ {
			zi += n.tUz[i][j] * hPrev[j]
			ri += n.tUr[i][j] * hPrev[j]
		}
		z[i] = sigmoid(zi)
		r[i] = sigmoid(ri)
	}
	for i := 0; i < gruHidden; i++ {
		ci := n.tBh[i]
		for j := 0; j < baseInputs; j++ {
			ci += n.tWh[i][j] * x[j]
		}
		for j := 0; j < gruHidden; j++ {
			ci += n.tUh[i][j] * (r[j] * hPrev[j])
		}
		cand := tanh(ci)
		hNew[i] = (1-z[i])*hPrev[i] + z[i]*cand
	}
	for a := 0; a < brainActions; a++ {
		s := n.tBq[a]
		for k := 0; k < gruHidden; k++ {
			s += n.tWq[a][k] * hNew[k]
		}
		q[a] = s
	}
	return
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

// remember додає перехід у кільцевий буфер досвіду.
//
// [GO: MUTEX] Під замком, бо буфер може бути СПІЛЬНИМ і в нього пишуть РІЗНІ
// горутини воркер-пулу (з паралельної фази). Секція крихітна → контенції майже
// нема. (У незалежному режимі замок завжди вільний → майже безкоштовний.)
func (n *Net) remember(t transition) {
	n.mu.Lock()
	if n.replay == nil {
		n.replay = make([]transition, qReplaySize)
	}
	n.replay[n.replayHead] = t
	n.replayHead = (n.replayHead + 1) % qReplaySize
	if n.replayHead == 0 {
		n.replayFull = true
	}
	n.mu.Unlock()
}

// replayLen — скільки переходів реально лежить у буфері.
func (n *Net) replayLen() int {
	if n.replayFull {
		return qReplaySize
	}
	return n.replayHead
}

// rememberSeq — [RNN] додає відрізок траєкторії в буфер послідовностей.
// Під тим самим mu, що й remember (буфер може бути спільним, пишуть різні горутини).
func (n *Net) rememberSeq(seq sequence) {
	n.mu.Lock()
	if n.seqReplay == nil {
		n.seqReplay = make([]sequence, seqReplaySize)
	}
	n.seqReplay[n.seqHead] = seq
	n.seqHead = (n.seqHead + 1) % seqReplaySize
	if n.seqHead == 0 {
		n.seqFull = true
	}
	n.mu.Unlock()
}

// seqReplayLen — скільки відрізків реально лежить у буфері.
func (n *Net) seqReplayLen() int {
	if n.seqFull {
		return seqReplaySize
	}
	return n.seqHead
}

// train — k оновлень на випадкових вибірках із буфера (серце DQN).
//
// Викликається ОДИН раз за кадр ОДНОПОТОКОВО (g.trainBrains, поза паралельною
// фазою) → запис ваг безпечний без локу. Зі спільним мозком уся колективна
// вибірка тренує ОДНУ мережу нормальним темпом (а не N×qBatch разів за кадр).
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

// trainSeq — [RNN] k оновлень на випадкових ВІДРІЗКАХ із буфера послідовностей.
func (n *Net) trainSeq(k int) {
	m := n.seqReplayLen()
	if m < seqMinReplay {
		return
	}
	for i := 0; i < k; i++ {
		n.tdUpdateSeq(n.seqReplay[rand.Intn(m)])
	}
}

// Step — один крок агента: оцінити минулу дію, обрати нову. Викликається щокадру
// в паралельній фазі (calcAcceleration). НЕ тренує мережу — лише кладе досвід у
// буфер; навчання відбувається раз/кадр однопотоково у g.trainBrains() (бо
// мережа може бути спільною: тренувати її N×qBatch/кадр було б і неправильно,
// і небезпечно для гонок).
func (b *Brain) Step(cur [baseInputs]float32, dist float32, hitWall bool) int {
	b.age++ // [3] для автоспаду ε

	// [RNN] Рекурентний шлях — окремий, щоб не чіпати робочий frame-stacking.
	if useGRU {
		return b.stepGRU(cur, dist, hitWall)
	}

	// [ПАМ'ЯТЬ] Склеюємо поточний кадр + історію → повний вхід мережі.
	// Поточний кадр — ПЕРШИЙ у стеку, тож whiskers лишаються на індексах 5..12
	// (тому maxWhisker/escapeAction/proximity-reward працюють без змін).
	stacked := b.buildStacked(cur)

	// [МЕТРИКИ ПАМʼЯТІ] Чи бачить агент гравця ЦЬОГО кадру (вхід visible = cur[13]).
	visible := cur[baseInputs-1] > 0.5

	// 1) Нагорода за попередню дію → перехід у (можливо спільний) буфер.
	if b.hasPrev {
		// [МЕТРИКИ ПАМʼЯТІ] Зміна дистанції (prevDist→dist) — наслідок дії, обраної
		// МИНУЛОГО кадру. Якщо тоді агент був СЛІПИЙ (prevVisible=false) — це
		// «сліпе рішення»; фіксуємо, чи він усе одно наблизився. Реактивний агент
		// без памʼяті наосліп ≈ випадковий; агент із памʼяттю тримає слід → частіше +.
		if !b.prevVisible {
			b.mBlindN++
			if dist < b.prevDist {
				b.mBlindClosed++
			}
		}
		// [RL: REWARD SHAPING]
		// Хижак: наблизився → +. [SELF-PLAY] Жертва (flee): інвертуємо — далі → +.
		sign := float32(1)
		if b.flee {
			sign = -1
		}
		reward := sign * (b.prevDist - dist) * rewardCloserScale
		if hitWall {
			reward += rewardWallHit // [1a] по факту удару
		}
		// [1b] плавний штраф за рух У БІК близької стіни: whisker напрямку, в який
		// пішли минулого кадру. Градієнт «тримай дистанцію» ще ДО зіткнення.
		reward += rewardNearWall * b.prevState[5+b.prevAction]

		b.lastReward = reward // [МЕТРИКИ] для середньої нагороди по рою
		b.net.remember(transition{s: b.prevState, a: b.prevAction, r: reward, s2: stacked})
	}

	// 2) [2] Anti-stuck. КЛЮЧОВЕ: якщо агент наближається до гравця — він НЕ
	//    застряг (хай навіть тернеться об стіну, productively ковзаючи вздовж неї).
	madeProgress := b.hasPrev && dist < b.prevDist-stuckProgressEps
	switch {
	case madeProgress:
		if b.stuckCounter > 0 {
			b.stuckCounter -= stuckDecay
		}
	case hitWall:
		b.stuckCounter += stuckHitInc
	case maxWhisker(cur) >= stuckNearWall:
		b.stuckCounter += stuckNoProgInc
	case b.stuckCounter > 0:
		b.stuckCounter -= stuckDecay
	}

	// 3) Обираємо дію. У «фрустрації» — НАПРАВЛЕНИЙ вихід у найвідкритіший бік
	//    (надійніше за випадковий смик), інакше ε-greedy.
	var action int
	switch {
	case b.frustration > 0:
		b.frustration--
		action = escapeAction(cur)
	case b.stuckCounter >= stuckLimit:
		b.frustration = frustrationFrames
		b.stuckCounter = 0
		b.markStuck = true // [СТИГМЕРГІЯ] офіційно застряг тут → лишити слід
		action = escapeAction(cur)
	default:
		action = b.selectAction(stacked)
	}

	b.prevState = stacked
	b.prevAction = action
	b.prevDist = dist
	b.prevVisible = visible // [МЕТРИКИ ПАМʼЯТІ] видимість на момент цього рішення
	b.hasPrev = true
	b.lastAction = action

	// [ПАМ'ЯТЬ] Раз на stackSkip кадрів записуємо поточний кадр в історію (зсув).
	// Індекс СКРІЗЬ змінна i (не літерал 0) — інакше при stackFrames=1 масив frames
	// має тип [0] і Go бракує константний frames[0] ще на компіляції.
	b.frameTick++
	if b.frameTick >= stackSkip {
		b.frameTick = 0
		for i := stackFrames - 2; i >= 0; i-- {
			if i > 0 {
				b.frames[i] = b.frames[i-1] // зсуваємо старі кадри назад
			} else {
				b.frames[i] = cur // найновіший кадр — у позицію 0
			}
		}
	}
	return action
}

// stepGRU — [RNN] крок агента з рекурентною памʼяттю. КРОК 2: forward + вибір дії
// + anti-stuck + reward + накопичення ВІДРІЗКА у буфер послідовностей. Навчання
// (BPTT) ще НЕ підключене (крок 3) — ваги GRU поки не міняються, рій діє випадково;
// але дані для навчання вже течуть у seqReplay, а метрики reward/blind оживають.
func (b *Brain) stepGRU(cur [baseInputs]float32, dist float32, hitWall bool) int {
	// Рекурентний forward: несемо власний стан b.h крізь кадри.
	q, hNew := b.net.forwardGRU(cur, b.h)
	b.h = hNew

	visible := cur[baseInputs-1] > 0.5

	// Нагорода за ПОПЕРЕДНЮ дію (та сама схема, що й у стек-шляху) → крок у відрізок.
	if b.hasPrev {
		// [МЕТРИКИ ПАМʼЯТІ] blind-chase (як у стек-шляху).
		if !b.prevVisible {
			b.mBlindN++
			if dist < b.prevDist {
				b.mBlindClosed++
			}
		}
		sign := float32(1)
		if b.flee {
			sign = -1
		}
		reward := sign * (b.prevDist - dist) * rewardCloserScale
		if hitWall {
			reward += rewardWallHit
		}
		reward += rewardNearWall * b.gruPrevX[5+b.prevAction]
		b.lastReward = reward

		// Записуємо завершений крок (x_{t-1}, a_{t-1}, r) у накопичувач відрізка.
		b.seqX[b.seqN] = b.gruPrevX
		b.seqA[b.seqN] = b.prevAction
		b.seqR[b.seqN] = reward
		b.seqN++
		if b.seqN == seqLen {
			// Відрізок повний → у спільний буфер (xEnd = поточний кадр для bootstrap).
			b.net.rememberSeq(sequence{x: b.seqX, a: b.seqA, r: b.seqR, xEnd: cur})
			b.seqN = 0
		}
	}

	// Anti-stuck — та сама сітка безпеки, що й у стек-режимі (по вусах кадру).
	madeProgress := b.hasPrev && dist < b.prevDist-stuckProgressEps
	switch {
	case madeProgress:
		if b.stuckCounter > 0 {
			b.stuckCounter -= stuckDecay
		}
	case hitWall:
		b.stuckCounter += stuckHitInc
	case maxWhisker(cur) >= stuckNearWall:
		b.stuckCounter += stuckNoProgInc
	case b.stuckCounter > 0:
		b.stuckCounter -= stuckDecay
	}

	var action int
	switch {
	case b.frustration > 0:
		b.frustration--
		action = escapeAction(cur)
	case b.stuckCounter >= stuckLimit:
		b.frustration = frustrationFrames
		b.stuckCounter = 0
		b.markStuck = true
		action = escapeAction(cur)
	default:
		action = b.selectFromQ(q)
	}

	b.gruPrevX = cur
	b.prevAction = action
	b.prevDist = dist
	b.prevVisible = visible
	b.hasPrev = true
	b.lastAction = action
	return action
}

// buildStacked склеює поточний (свіжий) кадр + історичні семпли в повний вхід
// мережі: [cur | frames[0] | frames[1] | ...]. Не мутує стан.
func (b *Brain) buildStacked(cur [baseInputs]float32) [brainInputs]float32 {
	var s [brainInputs]float32
	copy(s[0:baseInputs], cur[:])
	for f := 0; f < stackFrames-1; f++ {
		copy(s[(f+1)*baseInputs:(f+2)*baseInputs], b.frames[f][:])
	}
	return s
}

// stackSteady будує стек, повторюючи ОДИН кадр (усталене сприйняття) — зручно
// для проб/тестів, коли історія не важлива.
func stackSteady(cur [baseInputs]float32) [brainInputs]float32 {
	var s [brainInputs]float32
	for f := 0; f < stackFrames; f++ {
		copy(s[f*baseInputs:(f+1)*baseInputs], cur[:])
	}
	return s
}

// tdUpdate — навчання Q-LEARNING через TD (temporal-difference) помилку.
//
// [RL: РІВНЯННЯ БЕЛЛМАНА]
//
//	target = reward + γ · max_a' Q(nextState, a')
//	tdError = target - Q(s, a)
//
// Далі — backprop цієї помилки. ВАЖЛИВО: помилку має ЛИШЕ дія, яку реально
// зробили. "Semi-gradient": target вважаємо КОНСТАНТОЮ (по target-мережі).
// Пише ваги → викликається лише з train() (однопотокова фаза).
func (n *Net) tdUpdate(s [brainInputs]float32, a int, reward float32, s2 [brainInputs]float32) {
	// Ціль за Беллманом по TARGET-мережі (max Q наступного стану — як константа).
	q2 := n.forwardQTarget(s2)
	maxNext := q2[argmaxQ(q2)]
	target := clamp(reward+qGamma*maxNext, -qClip, qClip)

	// Поточна оцінка + активації прихованого шару (для backprop) — по ЖИВІЙ мережі.
	q1, h1, h2 := n.forwardQ(s)
	rawTD := target - q1[a]
	// [МЕТРИКИ] сира величина «здивування» + рівень Q (канарки навчання/розбіжності).
	n.mTDSum += float32(math.Abs(float64(rawTD)))
	n.mQSum += q1[argmaxQ(q1)]
	n.mTDN++
	// [DQN: ERROR CLIPPING] обмежуємо TD-помилку до [-1,1].
	tdErr := clamp(rawTD, -1, 1)

	// [BACKPROP крізь 3 шари] Похибку має лише вихід дії a (лінійний → похідна 1).
	// Спершу рахуємо ВСІ дельти (по СТАРИХ вагах), потім оновлюємо ваги.
	// hidden2: delta2[k] = tdErr · W3[a][k] · tanh'(h2[k])
	var delta2 [brainHidden2]float32
	for k := 0; k < brainHidden2; k++ {
		delta2[k] = tdErr * n.W3[a][k] * (1 - h2[k]*h2[k])
	}
	// hidden1: delta1[j] = (Σ_k delta2[k]·W2[k][j]) · tanh'(h1[j])
	var delta1 [brainHidden1]float32
	for j := 0; j < brainHidden1; j++ {
		var sum float32
		for k := 0; k < brainHidden2; k++ {
			sum += delta2[k] * n.W2[k][j]
		}
		delta1[j] = sum * (1 - h1[j]*h1[j])
	}

	// Оновлення ваг (від виходу до входу).
	for k := 0; k < brainHidden2; k++ { // вихід: лише рядок дії a
		n.W3[a][k] += qLearnRate * tdErr * h2[k]
	}
	n.B3[a] += qLearnRate * tdErr
	for k := 0; k < brainHidden2; k++ { // hidden2
		for j := 0; j < brainHidden1; j++ {
			n.W2[k][j] += qLearnRate * delta2[k] * h1[j]
		}
		n.B2[k] += qLearnRate * delta2[k]
	}
	for j := 0; j < brainHidden1; j++ { // hidden1
		for i := 0; i < brainInputs; i++ {
			n.W1[j][i] += qLearnRate * delta1[j] * s[i]
		}
		n.B1[j] += qLearnRate * delta1[j]
	}

	n.clipWeights()

	// [DQN] Періодично «заморожуємо» свіжі ваги в target-мережу.
	n.syncCounter++
	if n.syncCounter >= qTargetSync {
		n.syncTarget()
		n.syncCounter = 0
	}
}

// tdUpdateSeq — [RNN] навчання на одному ВІДРІЗКУ через BPTT (backprop through time).
//
// Три фази:
//  1. FORWARD (жива мережа): проганяємо GRU по кадрах від нульового стану,
//     кешуючи ворота z,r, кандидати cand і стан h на кожному кроці.
//  2. TARGET: окремо проганяємо ЗАМОРОЖЕНУ мережу → Беллман-ціль для кожного кроку
//     (target = r + γ·max_a Q_tgt(наступний стан)); TD-помилка = ціль − Q(дію).
//  3. BACKWARD: гортаємо градієнти НАЗАД у часі. Помилка на h кожного кроку =
//     (з виходу цього кроку) + (з рекурентного звʼязку наступного кроку). Ваги НЕ
//     міняємо під час проходу (semi-gradient) — накопичуємо й застосовуємо в кінці.
//
// [Спрощення 1-ї версії] replay стартує з h=0 (не з реального стану на момент
// збору) — «stored-state problem». Працює, хоч і неідеально; burn-in — на потім.
func (n *Net) tdUpdateSeq(seq sequence) {
	// --- Фаза 1: forward живої мережі з кешем ---
	var hArr [seqLen + 1][gruHidden]float32 // hArr[t] = h_{t-1}; hArr[0]=0
	var zc, rc, cc [seqLen][gruHidden]float32
	for t := 0; t < seqLen; t++ {
		hp := hArr[t]
		x := seq.x[t]
		var z, r [gruHidden]float32
		for i := 0; i < gruHidden; i++ {
			zi, ri := n.Bz[i], n.Br[i]
			for j := 0; j < baseInputs; j++ {
				zi += n.Wz[i][j] * x[j]
				ri += n.Wr[i][j] * x[j]
			}
			for j := 0; j < gruHidden; j++ {
				zi += n.Uz[i][j] * hp[j]
				ri += n.Ur[i][j] * hp[j]
			}
			z[i], r[i] = sigmoid(zi), sigmoid(ri)
		}
		for i := 0; i < gruHidden; i++ {
			ci := n.Bh[i]
			for j := 0; j < baseInputs; j++ {
				ci += n.Wh[i][j] * x[j]
			}
			for j := 0; j < gruHidden; j++ {
				ci += n.Uh[i][j] * (r[j] * hp[j])
			}
			cand := tanh(ci)
			cc[t][i] = cand
			hArr[t+1][i] = (1-z[i])*hp[i] + z[i]*cand
		}
		zc[t], rc[t] = z, r
	}

	// --- Фаза 2: target-ціль для кожного кроку ---
	// Проганяємо target-мережу; qTgt[s] = Q_tgt у стані s. Для кроку t bootstrap
	// бере наступний стан: s=t+1 (в межах відрізка) або xEnd (останній крок).
	var hT [gruHidden]float32
	var qTgt [seqLen][brainActions]float32
	for t := 0; t < seqLen; t++ {
		qTgt[t], hT = n.forwardGRUTarget(seq.x[t], hT)
	}
	qEnd, _ := n.forwardGRUTarget(seq.xEnd, hT)

	// TD-помилка на кожному кроці (semi-gradient: ціль — константа).
	var td [seqLen]float32
	for t := 0; t < seqLen; t++ {
		var qNext [brainActions]float32
		if t < seqLen-1 {
			qNext = qTgt[t+1]
		} else {
			qNext = qEnd
		}
		target := clamp(seq.r[t]+qGamma*qNext[argmaxQ(qNext)], -qClip, qClip)

		// Q(дію) живої мережі зі стану h_t (= hArr[t+1]).
		a := seq.a[t]
		qa := n.Bq[a]
		for k := 0; k < gruHidden; k++ {
			qa += n.Wq[a][k] * hArr[t+1][k]
		}
		rawTD := target - qa
		n.mTDSum += float32(math.Abs(float64(rawTD)))
		n.mQSum += qa
		n.mTDN++
		td[t] = clamp(rawTD, -1, 1)
	}

	// --- Фаза 3: BPTT (градієнти в акумулятори, застосовуємо в кінці) ---
	var dWz, dWr, dWh [gruHidden][baseInputs]float32
	var dUz, dUr, dUh [gruHidden][gruHidden]float32
	var dBz, dBr, dBh [gruHidden]float32
	var dWq [brainActions][gruHidden]float32
	var dBq [brainActions]float32

	var dhNext [gruHidden]float32 // градієнт, що тече з майбутнього кроку в h_t
	for t := seqLen - 1; t >= 0; t-- {
		hp := hArr[t]
		ht := hArr[t+1]
		x := seq.x[t]
		a := seq.a[t]
		z, r, cand := zc[t], rc[t], cc[t]

		// Помилка на h_t = з виходу (лише дія a) + з рекурентного звʼязку.
		var g [gruHidden]float32
		for k := 0; k < gruHidden; k++ {
			g[k] = td[t]*n.Wq[a][k] + dhNext[k]
		}
		for k := 0; k < gruHidden; k++ { // градієнт виходу
			dWq[a][k] += td[t] * ht[k]
		}
		dBq[a] += td[t]

		// Через клітину: спершу gzin і gcin (потрібні для reset-градієнта).
		var gzin, gcin [gruHidden]float32
		for i := 0; i < gruHidden; i++ {
			gcin[i] = g[i] * z[i] * (1 - cand[i]*cand[i])          // через кандидат
			gzin[i] = g[i] * (cand[i] - hp[i]) * z[i] * (1 - z[i]) // через update-ворота
		}
		// reset-ворота: r_j впливає на ВСІ кандидати → сума по i.
		var grin [gruHidden]float32
		for j := 0; j < gruHidden; j++ {
			var gr float32
			for i := 0; i < gruHidden; i++ {
				gr += gcin[i] * n.Uh[i][j] * hp[j]
			}
			grin[j] = gr * r[j] * (1 - r[j])
		}

		// Накопичуємо градієнти ваг.
		for i := 0; i < gruHidden; i++ {
			for m := 0; m < baseInputs; m++ {
				dWz[i][m] += gzin[i] * x[m]
				dWr[i][m] += grin[i] * x[m]
				dWh[i][m] += gcin[i] * x[m]
			}
			for j := 0; j < gruHidden; j++ {
				dUz[i][j] += gzin[i] * hp[j]
				dUr[i][j] += grin[i] * hp[j]
				dUh[i][j] += gcin[i] * (r[j] * hp[j])
			}
			dBz[i] += gzin[i]
			dBr[i] += grin[i]
			dBh[i] += gcin[i]
		}

		// Градієнт у h_{t-1} (для наступної ітерації назад): 4 шляхи.
		var dhPrev [gruHidden]float32
		for j := 0; j < gruHidden; j++ {
			s := g[j] * (1 - z[j]) // прямий шлях (1-z)⊙hp
			for i := 0; i < gruHidden; i++ {
				s += gzin[i] * n.Uz[i][j]
				s += grin[i] * n.Ur[i][j]
				s += gcin[i] * n.Uh[i][j] * r[j]
			}
			dhPrev[j] = clamp(s, -gruGradClip, gruGradClip) // кліп проти вибуху в часі
		}
		dhNext = dhPrev
	}

	// --- Застосування (крок ГРАДІЄНТНОГО ПІДЙОМУ на +tdErr, як у стек-tdUpdate) ---
	cg := func(v float32) float32 { return clamp(v, -gruGradClip, gruGradClip) }
	for i := 0; i < gruHidden; i++ {
		for m := 0; m < baseInputs; m++ {
			n.Wz[i][m] += qLearnRate * cg(dWz[i][m])
			n.Wr[i][m] += qLearnRate * cg(dWr[i][m])
			n.Wh[i][m] += qLearnRate * cg(dWh[i][m])
		}
		for j := 0; j < gruHidden; j++ {
			n.Uz[i][j] += qLearnRate * cg(dUz[i][j])
			n.Ur[i][j] += qLearnRate * cg(dUr[i][j])
			n.Uh[i][j] += qLearnRate * cg(dUh[i][j])
		}
		n.Bz[i] += qLearnRate * cg(dBz[i])
		n.Br[i] += qLearnRate * cg(dBr[i])
		n.Bh[i] += qLearnRate * cg(dBh[i])
	}
	for a := 0; a < brainActions; a++ {
		for k := 0; k < gruHidden; k++ {
			n.Wq[a][k] += qLearnRate * cg(dWq[a][k])
		}
		n.Bq[a] += qLearnRate * cg(dBq[a])
	}

	n.clipGRU()

	n.syncCounter++
	if n.syncCounter >= qTargetSync {
		n.syncTarget()
		n.syncCounter = 0
	}
}

// clipWeights обрізає всі ваги до [-brainMaxWeight, +brainMaxWeight].
func (n *Net) clipWeights() {
	for j := range n.W1 {
		for i := range n.W1[j] {
			n.W1[j][i] = clamp(n.W1[j][i], -brainMaxWeight, brainMaxWeight)
		}
		n.B1[j] = clamp(n.B1[j], -brainMaxWeight, brainMaxWeight)
	}
	for k := range n.W2 {
		for j := range n.W2[k] {
			n.W2[k][j] = clamp(n.W2[k][j], -brainMaxWeight, brainMaxWeight)
		}
		n.B2[k] = clamp(n.B2[k], -brainMaxWeight, brainMaxWeight)
	}
	for a := range n.W3 {
		for k := range n.W3[a] {
			n.W3[a][k] = clamp(n.W3[a][k], -brainMaxWeight, brainMaxWeight)
		}
		n.B3[a] = clamp(n.B3[a], -brainMaxWeight, brainMaxWeight)
	}
}

// clipGRU обрізає ваги рекурентної клітини до [-brainMaxWeight, +brainMaxWeight].
func (n *Net) clipGRU() {
	c := func(v float32) float32 { return clamp(v, -brainMaxWeight, brainMaxWeight) }
	for i := 0; i < gruHidden; i++ {
		for m := 0; m < baseInputs; m++ {
			n.Wz[i][m], n.Wr[i][m], n.Wh[i][m] = c(n.Wz[i][m]), c(n.Wr[i][m]), c(n.Wh[i][m])
		}
		for j := 0; j < gruHidden; j++ {
			n.Uz[i][j], n.Ur[i][j], n.Uh[i][j] = c(n.Uz[i][j]), c(n.Ur[i][j]), c(n.Uh[i][j])
		}
		n.Bz[i], n.Br[i], n.Bh[i] = c(n.Bz[i]), c(n.Br[i]), c(n.Bh[i])
	}
	for a := 0; a < brainActions; a++ {
		for k := 0; k < gruHidden; k++ {
			n.Wq[a][k] = c(n.Wq[a][k])
		}
		n.Bq[a] = c(n.Bq[a])
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
