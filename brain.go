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
//   state(13) ─► hidden(16, tanh) ─► Q-values(8, лінійні)
//   13 входів = 5 базових (dx,dy,dist,pVelX,pVelY) + 8 whiskers (сенсори стін)
//   8 виходів = Q-значення для 8 напрямків руху. Дія = напрямок з найбільшим Q.
//
// [SHARED BRAIN] Поділ на Net + Brain:
//   Net   — сама мережа (ваги + target + буфер досвіду). Може бути СПІЛЬНОЮ.
//   Brain — «голова» одного ворога: указник на Net + ОСОБИСТА пам'ять агента.
//   Режим sharedBrain (main.go): усі учні ділять один Net → «вулик-розум».
// ==========================================================================

const (
	brainInputs   = 13 // 5 базових + 8 whiskers
	brainHidden   = 16 // нейрони прихованого шару
	brainActions  = 8  // 8 напрямків руху (= кількість виходів Q)
	brainWhiskers = 8  // промені-сенсори стін

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
// Це «одиниця пам'яті» для experience replay.
type transition struct {
	s  [brainInputs]float32
	a  int
	r  float32
	s2 [brainInputs]float32
}

// Net — НЕЙРОМЕРЕЖА Q-агента: ваги + target-копія + буфер досвіду.
//
// Кілька Brain можуть указувати на ОДИН Net (режим sharedBrain=true) → «вулик-
// розум»: усі ділять одну вивчену політику й спільний досвід.
//
// [GO: ВАГИ ДВОХ ШАРІВ] W1[j][i] — вхід i → прихований j; W2[k][j] — j → вихід k.
type Net struct {
	W1 [brainHidden][brainInputs]float32
	B1 [brainHidden]float32
	W2 [brainActions][brainHidden]float32
	B2 [brainActions]float32

	// [DQN: TARGET NETWORK] заморожена копія для Беллман-цілі (щоб не «тікала»).
	tW1         [brainHidden][brainInputs]float32
	tB1         [brainHidden]float32
	tW2         [brainActions][brainHidden]float32
	tB2         [brainActions]float32
	syncCounter int

	// [DQN: EXPERIENCE REPLAY] кільцевий буфер переходів.
	replay     []transition
	replayHead int
	replayFull bool

	// [GO: MUTEX] захищає СПІЛЬНИЙ буфер від одночасного запису з різних горутин
	// (remember у паралельній фазі calcAcceleration). Ваги ж безпечні без локу
	// через РОЗДІЛЕННЯ ФАЗ: forward читається паралельно, train пише однопотоково
	// (g.trainBrains) — фази не перетинаються.
	mu sync.Mutex
}

// Brain — «голова» одного ворога-учня: указник на мережу + ОСОБИСТА пам'ять.
// Мережа може бути спільною; пам'ять (стан у часі, лічильники) — завжди своя,
// тож агенти діють індивідуально, але вчаться в (можливо) спільну мережу.
type Brain struct {
	net *Net

	// [RL: ПАМ'ЯТЬ МІЖ КАДРАМИ] — у кожного агента своя.
	// Reward за дію відомий лише НАСТУПНОГО кадру (коли побачимо результат руху).
	prevState  [brainInputs]float32
	prevAction int
	prevDist   float32
	hasPrev    bool

	age int // [3] вік (к-сть Step) — для автоспаду ε

	// [2] anti-stuck: лічильник застрягання, залишок кадрів «фрустрації», прапорець сліду.
	stuckCounter int
	frustration  int
	markStuck    bool

	// Для візуалізації (читає Draw, пише calcAcceleration — різні фази, без гонки).
	lastWhiskers [brainWhiskers]float32
	lastAction   int
}

// tanh — активація прихованого шару. Похідна: tanh'(z) = 1 - tanh(z)².
func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
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
	s2 := float32(math.Sqrt(1.0 / brainHidden))
	for k := range n.W2 {
		for j := range n.W2[k] {
			n.W2[k][j] = (rand.Float32()*2 - 1) * s2
		}
	}
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
	n.tW1, n.tB1, n.tW2, n.tB2 = n.W1, n.B1, n.W2, n.B2
}

// forwardQTarget — як forwardQ, але по ЗАМОРОЖЕНИХ (target) вагах.
// Використовується лише для обрахунку Беллман-цілі.
func (n *Net) forwardQTarget(state [brainInputs]float32) [brainActions]float32 {
	var hidden [brainHidden]float32
	for j := 0; j < brainHidden; j++ {
		z := n.tB1[j]
		for i := 0; i < brainInputs; i++ {
			z += state[i] * n.tW1[j][i]
		}
		hidden[j] = tanh(z)
	}
	var q [brainActions]float32
	for k := 0; k < brainActions; k++ {
		z := n.tB2[k]
		for j := 0; j < brainHidden; j++ {
			z += hidden[j] * n.tW2[k][j]
		}
		q[k] = z
	}
	return q
}

// forwardQ — прямий прохід: state → Q-значення всіх 8 дій.
//
// [RL: Q(s, a) = "наскільки хороша дія a у стані s"]
// Q ≈ очікувана сумарна майбутня нагорода, якщо зробити a, а далі діяти жадібно.
//
// Прихований шар з tanh (нелінійність), вихід ЛІНІЙНИЙ — бо Q-значення можуть
// бути будь-якими числами. Функція ЧИСТА (лише читає ваги) → безпечна для
// паралельного виклику, поки ніхто не ПИШЕ ваги (а пишемо ми лише в trainBrains).
func (n *Net) forwardQ(state [brainInputs]float32) (q [brainActions]float32, hidden [brainHidden]float32) {
	for j := 0; j < brainHidden; j++ {
		z := n.B1[j]
		for i := 0; i < brainInputs; i++ {
			z += state[i] * n.W1[j][i]
		}
		hidden[j] = tanh(z)
	}
	for k := 0; k < brainActions; k++ {
		z := n.B2[k]
		for j := 0; j < brainHidden; j++ {
			z += hidden[j] * n.W2[k][j]
		}
		q[k] = z // лінійний вихід
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
func maxWhisker(state [brainInputs]float32) float32 {
	m := state[5]
	for i := 1; i < brainWhiskers; i++ {
		if state[5+i] > m {
			m = state[5+i]
		}
	}
	return m
}

// escapeAction — напрямок із НАЙМЕНШОЮ близькістю стіни (найвідкритіший), щоб
// гарантовано вийти з пастки, а не смикатись на місці. Серед однаково відкритих
// напрямків — рівноймовірно (reservoir), аби агенти не злипались в один бік.
func escapeAction(state [brainInputs]float32) int {
	best, bestW, ties := 0, state[5], 1
	for i := 1; i < brainWhiskers; i++ {
		w := state[5+i]
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
	q, _ := b.net.forwardQ(state)
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

// train — k оновлень на випадкових вибірках із буфера (серце DQN).
//
// Викликається ОДИН раз за кадр ОДНОПОТОКОВО (g.trainBrains, поза паралельною
// фазою) → запис ваг безпечний без локу. Зі спільним мозком уся колективна
// вибірка тренує ОДНУ мережу нормальним темпом (а не N×qBatch разів за кадр).
func (n *Net) train(k int) {
	m := n.replayLen()
	if m < qMinReplay {
		return
	}
	for i := 0; i < k; i++ {
		t := n.replay[rand.Intn(m)]
		n.tdUpdate(t.s, t.a, t.r, t.s2)
	}
}

// Step — один крок агента: оцінити минулу дію, обрати нову. Викликається щокадру
// в паралельній фазі (calcAcceleration). НЕ тренує мережу — лише кладе досвід у
// буфер; навчання відбувається раз/кадр однопотоково у g.trainBrains() (бо
// мережа може бути спільною: тренувати її N×qBatch/кадр було б і неправильно,
// і небезпечно для гонок).
func (b *Brain) Step(state [brainInputs]float32, dist float32, hitWall bool) int {
	b.age++ // [3] для автоспаду ε

	// 1) Нагорода за попередню дію → перехід у (можливо спільний) буфер.
	if b.hasPrev {
		// [RL: REWARD SHAPING]
		// Щільна нагорода веде агента: наблизився → +, віддалився → −.
		reward := (b.prevDist - dist) * rewardCloserScale
		if hitWall {
			reward += rewardWallHit // [1a] по факту удару
		}
		// [1b] плавний штраф за рух У БІК близької стіни: whisker напрямку, в який
		// пішли минулого кадру. Градієнт «тримай дистанцію» ще ДО зіткнення.
		reward += rewardNearWall * b.prevState[5+b.prevAction]

		b.net.remember(transition{s: b.prevState, a: b.prevAction, r: reward, s2: state})
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
	case maxWhisker(state) >= stuckNearWall:
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
		action = escapeAction(state)
	case b.stuckCounter >= stuckLimit:
		b.frustration = frustrationFrames
		b.stuckCounter = 0
		b.markStuck = true // [СТИГМЕРГІЯ] офіційно застряг тут → лишити слід
		action = escapeAction(state)
	default:
		action = b.selectAction(state)
	}

	b.prevState = state
	b.prevAction = action
	b.prevDist = dist
	b.hasPrev = true
	b.lastAction = action
	return action
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
	q1, hidden := n.forwardQ(s)
	// [DQN: ERROR CLIPPING] обмежуємо TD-помилку до [-1,1].
	tdErr := clamp(target-q1[a], -1, 1)

	// Вихідний шар: похибку має лише нейрон дії a (лінійний вихід → похідна 1).
	// Прихований шар: проштовхуємо похибку назад через W2[a].
	for j := 0; j < brainHidden; j++ {
		// deltaHidden рахуємо ДО оновлення W2[a][j] (по старій вазі).
		deltaHidden := tdErr * n.W2[a][j] * (1 - hidden[j]*hidden[j])
		n.W2[a][j] += qLearnRate * tdErr * hidden[j]
		for i := 0; i < brainInputs; i++ {
			n.W1[j][i] += qLearnRate * deltaHidden * s[i]
		}
		n.B1[j] += qLearnRate * deltaHidden
	}
	n.B2[a] += qLearnRate * tdErr

	n.clipWeights()

	// [DQN] Періодично «заморожуємо» свіжі ваги в target-мережу.
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

// GatherInputs збирає стан (state) для Q-мережі.
//
//	[0] dx/screenWidth     напрямок до гравця X
//	[1] dy/screenHeight    напрямок до гравця Y
//	[2] dist/screenWidth   відстань до гравця
//	[3] pVelX/5            швидкість гравця X
//	[4] pVelY/5            швидкість гравця Y
//	[5..12] whiskers       близькість стіни у 8 напрямках  ◄── зір на перешкоди
//
// Побічно зберігає whiskers у Brain для візуалізації.
func GatherInputs(enemy, player *Pixel) [brainInputs]float32 {
	dx := player.X - enemy.X
	dy := player.Y - enemy.Y
	dist := float32(math.Sqrt(float64(dx*dx + dy*dy)))
	if dist == 0 {
		dist = 1
	}

	var in [brainInputs]float32
	in[0] = dx / screenWidth
	in[1] = dy / screenHeight
	in[2] = dist / screenWidth
	in[3] = player.VelX / 5.0
	in[4] = player.VelY / 5.0

	cx := enemy.X + pixelSize/2
	cy := enemy.Y + pixelSize/2
	for i := 0; i < brainWhiskers; i++ {
		w := wallWhisker(cx, cy, dirs8[i][0], dirs8[i][1])
		in[5+i] = w
		if enemy.Brain != nil {
			enemy.Brain.lastWhiskers[i] = w
		}
	}
	return in
}

// brainFile — шлях до файлу де зберігаються вивчені ваги між сесіями.
const brainFile = "brain_weights.json"

// BrainData — серіалізація ваг у JSON + розміри мережі для перевірки сумісності.
type BrainData struct {
	Inputs  int `json:"inputs"`
	Hidden  int `json:"hidden"`
	Actions int `json:"actions"`

	W1 [brainHidden][brainInputs]float32  `json:"w1"`
	B1 [brainHidden]float32               `json:"b1"`
	W2 [brainActions][brainHidden]float32 `json:"w2"`
	B2 [brainActions]float32              `json:"b2"`
}

// SaveBrain зберігає ваги мережі агента у JSON.
func SaveBrain(b *Brain) error { return SaveNet(b.net) }

// SaveNet зберігає ваги мережі у JSON (читабельний MarshalIndent).
func SaveNet(n *Net) error {
	data := BrainData{
		Inputs: brainInputs, Hidden: brainHidden, Actions: brainActions,
		W1: n.W1, B1: n.B1, W2: n.W2, B2: n.B2,
	}
	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(brainFile, bytes, 0644)
}

// LoadNet завантажує мережу з файлу. Повертає nil (→ caller створить NewNet),
// якщо файлу немає, він пошкоджений, або РОЗМІРИ мережі не збігаються (зміна
// архітектури). Перевірка dims рятує від часткового завантаження.
func LoadNet() *Net {
	bytes, err := os.ReadFile(brainFile)
	if err != nil {
		return nil
	}
	var data BrainData
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil
	}
	if data.Inputs != brainInputs || data.Hidden != brainHidden || data.Actions != brainActions {
		return nil // несумісна архітектура → почнемо з нуля
	}
	n := &Net{W1: data.W1, B1: data.B1, W2: data.W2, B2: data.B2}
	n.syncTarget()
	return n
}
