package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
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

// Brain — Q-мережа одного ворога-учня + його пам'ять для RL.
//
// [GO: ВАГИ ДВОХ ШАРІВ]
// W1[j][i] — вхід i → прихований нейрон j;  W2[k][j] — прихований j → Q-вихід k.
type Brain struct {
	W1 [brainHidden][brainInputs]float32 // input → hidden
	B1 [brainHidden]float32
	W2 [brainActions][brainHidden]float32 // hidden → Q-values
	B2 [brainActions]float32

	// [DQN: TARGET NETWORK]
	// Заморожена копія ваг. Беллман-ціль рахуємо ПО НІЙ, а не по живій мережі.
	// Так ціль не «тікає» з кожним кроком (жива і target розв'язані) — навчання
	// не женеться за власним хвостом. Раз на qTargetSync кроків копіюємо живі ваги.
	tW1         [brainHidden][brainInputs]float32
	tB1         [brainHidden]float32
	tW2         [brainActions][brainHidden]float32
	tB2         [brainActions]float32
	syncCounter int

	// [RL: ПАМ'ЯТЬ МІЖ КАДРАМИ]
	// Щоб навчатись на переході (state, action, reward, nextState), треба
	// пам'ятати, що було минулого кадру. Reward за дію стає відомий лише
	// НАСТУПНОГО кадру (коли побачимо результат руху).
	prevState  [brainInputs]float32
	prevAction int
	prevDist   float32
	hasPrev    bool

	// [3] вік мозку (к-сть викликів Step) — для автоспаду ε.
	age int

	// [2] anti-stuck: лічильник упертого биття в стіну і залишок кадрів «фрустрації».
	stuckCounter int
	frustration  int
	markStuck    bool // [СТИГМЕРГІЯ] щойно «офіційно» застряг → лишити слід у клітинці

	// [DQN: EXPERIENCE REPLAY]
	// Кільцевий буфер минулих переходів. Замість навчання на свіжому (і сильно
	// корельованому з попереднім) переході — щокадру беремо ВИПАДКОВУ вибірку
	// зі старих. Це розриває кореляцію сусідніх кадрів і прибирає
	// «catastrophic forgetting» (коли мережа забуває старе, переучуючись на нове).
	replay     []transition
	replayHead int
	replayFull bool

	// Для візуалізації (читає Draw, пише calcAcceleration — різні фази, без гонки).
	lastWhiskers [brainWhiskers]float32
	lastAction   int
}

// transition — один крок досвіду: (стан, дія, нагорода, наступний стан).
// Це «одиниця пам'яті» для experience replay.
type transition struct {
	s  [brainInputs]float32
	a  int
	r  float32
	s2 [brainInputs]float32
}

// tanh — активація прихованого шару. Похідна: tanh'(z) = 1 - tanh(z)².
func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// NewBrain створює Q-мережу з Xavier-ініціалізацією (масштаб ~1/√fan_in),
// щоб tanh не входив у насичення і градієнт не зникав.
func NewBrain() *Brain {
	b := &Brain{}
	s1 := float32(math.Sqrt(1.0 / brainInputs))
	for j := range b.W1 {
		for i := range b.W1[j] {
			b.W1[j][i] = (rand.Float32()*2 - 1) * s1
		}
	}
	s2 := float32(math.Sqrt(1.0 / brainHidden))
	for k := range b.W2 {
		for j := range b.W2[k] {
			b.W2[k][j] = (rand.Float32()*2 - 1) * s2
		}
	}
	b.syncTarget() // target стартує копією живих ваг
	return b
}

// syncTarget копіює живі ваги в target-мережу.
// [GO: масиви — значимі типи] присвоєння масиву копіює його повністю.
func (b *Brain) syncTarget() {
	b.tW1, b.tB1, b.tW2, b.tB2 = b.W1, b.B1, b.W2, b.B2
}

// forwardQTarget — як forwardQ, але по ЗАМОРОЖЕНИХ (target) вагах.
// Використовується лише для обрахунку Беллман-цілі.
func (b *Brain) forwardQTarget(state [brainInputs]float32) [brainActions]float32 {
	var hidden [brainHidden]float32
	for j := 0; j < brainHidden; j++ {
		z := b.tB1[j]
		for i := 0; i < brainInputs; i++ {
			z += state[i] * b.tW1[j][i]
		}
		hidden[j] = tanh(z)
	}
	var q [brainActions]float32
	for k := 0; k < brainActions; k++ {
		z := b.tB2[k]
		for j := 0; j < brainHidden; j++ {
			z += hidden[j] * b.tW2[k][j]
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
// бути будь-якими числами (не обмежені -1..+1, як був вихід у v2).
// Функція ЧИСТА (нічого не змінює) → безпечно кликати кілька разів за кадр.
func (b *Brain) forwardQ(state [brainInputs]float32) (q [brainActions]float32, hidden [brainHidden]float32) {
	for j := 0; j < brainHidden; j++ {
		z := b.B1[j]
		for i := 0; i < brainInputs; i++ {
			z += state[i] * b.W1[j][i]
		}
		hidden[j] = tanh(z)
	}
	for k := 0; k < brainActions; k++ {
		z := b.B2[k]
		for j := 0; j < brainHidden; j++ {
			z += hidden[j] * b.W2[k][j]
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

// selectAction — ε-GREEDY вибір дії.
//
// [RL: EXPLORATION vs EXPLOITATION]
// З імовірністю ε — випадкова дія (досліджуємо світ, шукаємо нові стратегії).
// Інакше — найкраща за Q (використовуємо вивчене). Без exploration агент
// застрягне на першій-ліпшій стратегії й не знайде кращої.
func (b *Brain) selectAction(state [brainInputs]float32) int {
	if rand.Float32() < b.epsilon() {
		return rand.Intn(brainActions)
	}
	q, _ := b.forwardQ(state)
	return argmaxQ(q)
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

// remember додає перехід у кільцевий буфер досвіду.
func (b *Brain) remember(t transition) {
	if b.replay == nil {
		b.replay = make([]transition, qReplaySize)
	}
	b.replay[b.replayHead] = t
	b.replayHead = (b.replayHead + 1) % qReplaySize
	if b.replayHead == 0 {
		b.replayFull = true
	}
}

// replayLen — скільки переходів реально лежить у буфері.
func (b *Brain) replayLen() int {
	if b.replayFull {
		return qReplaySize
	}
	return b.replayHead
}

// trainBatch — навчання на випадковій вибірці з буфера (серце DQN).
func (b *Brain) trainBatch() {
	n := b.replayLen()
	if n < qMinReplay {
		return
	}
	for i := 0; i < qBatch; i++ {
		t := b.replay[rand.Intn(n)]
		b.tdUpdate(t.s, t.a, t.r, t.s2)
	}
}

// Step — один крок агента: оцінити минулу дію, обрати нову.
//
// Викликається щокадру. Reward за ПОПЕРЕДНЮ дію тепер відомий (бачимо результат
// руху: dist змінилась, можливо врізались у стіну) → кладемо перехід у буфер
// і вчимось на випадковій вибірці зі ВСЬОГО накопиченого досвіду.
func (b *Brain) Step(state [brainInputs]float32, dist float32, hitWall bool) int {
	b.age++ // [3] для автоспаду ε

	// 1) Нагорода за попередню дію → перехід у буфер → навчання на вибірці.
	if b.hasPrev {
		// [RL: REWARD SHAPING]
		// Щільна нагорода веде агента: наблизився → +, віддалився → −.
		reward := (b.prevDist - dist) * rewardCloserScale
		if hitWall {
			reward += rewardWallHit // [1a] по факту удару
		}
		// [1b] плавний штраф за рух У БІК близької стіни: беремо whisker того
		// напрямку, в який пішли минулого кадру (prevState[5+prevAction]). Дає
		// градієнт «тримай дистанцію» ще ДО зіткнення → обхід стає плавним.
		// Прогрес до гравця (+0.5) перебиває цей штраф у проходах, тож щілини
		// агент усе одно використовує — уникає лише глухих стін.
		reward += rewardNearWall * b.prevState[5+b.prevAction]

		b.remember(transition{s: b.prevState, a: b.prevAction, r: reward, s2: state})
		b.trainBatch()
	}

	// 2) [2] Anti-stuck. КЛЮЧОВЕ: якщо агент наближається до гравця — він НЕ
	//    застряг (хай навіть тернеться об стіну, productively ковзаючи вздовж неї).
	//    Лічильник росте лише коли НЕМАЄ прогресу І є контакт/близькість стіни.
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
// Цінність дії = миттєва нагорода + найкраще, що можна отримати далі:
//
//	target = reward + γ · max_a' Q(nextState, a')
//
// Це розв'язує "credit assignment" (хто винен за відкладену нагороду):
// цінність ПРОСОЧУЄТЬСЯ назад у часі — кадр за кадром, через γ.
//
// TD-помилка = наскільки наша оцінка Q(s,a) розходиться з target:
//
//	tdError = target - Q(s, a)
//
// Далі — звичайний backprop цієї помилки. ВАЖЛИВО: помилку має ЛИШЕ дія, яку
// реально зробили (тільки про неї ми отримали reward); інші виходи не чіпаємо.
//
// "Semi-gradient": target вважаємо КОНСТАНТОЮ (не пускаємо градієнт у Q(s')) —
// інакше навчання женеться за власним хвостом і розходиться.
func (b *Brain) tdUpdate(s [brainInputs]float32, a int, reward float32, s2 [brainInputs]float32) {
	// Ціль за Беллманом по TARGET-мережі (max Q наступного стану — як константа).
	q2 := b.forwardQTarget(s2)
	maxNext := q2[argmaxQ(q2)]
	target := clamp(reward+qGamma*maxNext, -qClip, qClip)

	// Поточна оцінка + активації прихованого шару (для backprop) — по ЖИВІЙ мережі.
	q1, hidden := b.forwardQ(s)
	// [DQN: ERROR CLIPPING] обмежуємо TD-помилку до [-1,1] → жодних велетенських
	// стрибків ваг від рідкісних великих похибок (Huber-подібна стабілізація).
	tdErr := clamp(target-q1[a], -1, 1)

	// Вихідний шар: похибку має лише нейрон дії a (лінійний вихід → похідна 1).
	//   W2[a][j] += lr · tdErr · hidden[j]
	// Прихований шар: проштовхуємо похибку назад через W2[a] (тільки цей рядок
	// бере участь, бо лише вихід a має ненульову похибку).
	for j := 0; j < brainHidden; j++ {
		// deltaHidden рахуємо ДО оновлення W2[a][j] (по старій вазі).
		deltaHidden := tdErr * b.W2[a][j] * (1 - hidden[j]*hidden[j])
		b.W2[a][j] += qLearnRate * tdErr * hidden[j]
		for i := 0; i < brainInputs; i++ {
			b.W1[j][i] += qLearnRate * deltaHidden * s[i]
		}
		b.B1[j] += qLearnRate * deltaHidden
	}
	b.B2[a] += qLearnRate * tdErr

	b.clipWeights()

	// [DQN] Періодично «заморожуємо» свіжі ваги в target-мережу.
	b.syncCounter++
	if b.syncCounter >= qTargetSync {
		b.syncTarget()
		b.syncCounter = 0
	}
}

// clipWeights обрізає всі ваги до [-brainMaxWeight, +brainMaxWeight].
func (b *Brain) clipWeights() {
	for j := range b.W1 {
		for i := range b.W1[j] {
			b.W1[j][i] = clamp(b.W1[j][i], -brainMaxWeight, brainMaxWeight)
		}
		b.B1[j] = clamp(b.B1[j], -brainMaxWeight, brainMaxWeight)
	}
	for k := range b.W2 {
		for j := range b.W2[k] {
			b.W2[k][j] = clamp(b.W2[k][j], -brainMaxWeight, brainMaxWeight)
		}
		b.B2[k] = clamp(b.B2[k], -brainMaxWeight, brainMaxWeight)
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
// Так мережа отримує "зір" на перешкоди ще до зіткнення.
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

// SaveBrain зберігає ваги у JSON (читабельний MarshalIndent).
func SaveBrain(b *Brain) error {
	data := BrainData{
		Inputs: brainInputs, Hidden: brainHidden, Actions: brainActions,
		W1: b.W1, B1: b.B1, W2: b.W2, B2: b.B2,
	}
	bytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(brainFile, bytes, 0644)
}

// LoadBrain завантажує ваги. Повертає nil (→ caller створить NewBrain), якщо
// файлу немає, він пошкоджений, або РОЗМІРИ мережі не збігаються (зміна
// архітектури між версіями). Перевірка dims рятує від часткового завантаження.
func LoadBrain() *Brain {
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
	b := &Brain{W1: data.W1, B1: data.B1, W2: data.W2, B2: data.B2}
	b.syncTarget()
	b.age = qEpsilonDecay // завантажений = вже навчений → старт на ε-floor (режим мисливця)
	return b
}
