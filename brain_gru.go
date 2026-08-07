package main

import (
	"math"
	"math/rand"
)

// ==========================================================================
// ШЛЯХ 2: GRU — ВИВЧЕНА памʼять (рекурентна) — активний при useGRU = true.
//
// На вхід іде ОДИН кадр (baseInputs), а «минуле» живе в прихованому стані h,
// який мережа несе між кадрами й САМА вирішує (воротами z/r), що тримати і як
// довго. Немає фіксованого вікна, як у стеку.
//
// Навчання — BPTT на ВІДРІЗКАХ траєкторії (sequence-replay) з burn-in-прогрівом.
//
// Стек-шлях — у brain_stack.go. Спільне ядро — у brain.go.
// ==========================================================================

// sequence — [RNN] відрізок траєкторії ОДНОГО агента: seqTotal поспіль кадрів
// (вхід x, дія a, нагорода r) + xEnd (кадр ПІСЛЯ останнього, для bootstrap).
// Перші seqBurnIn кадрів — ПРОГРІВ (forward-only, щоб h став реальним), решта
// seqLen — навчальні (loss + BPTT). Це «одиниця пам'яті» рекурентного навчання.
// (a/r для burn-in кадрів зберігаємо, але в навчанні не вживаємо.)
type sequence struct {
	x    [seqTotal][baseInputs]float32
	a    [seqTotal]int
	r    [seqTotal]float32
	xEnd [baseInputs]float32
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

// stepGRU — [RNN] крок агента з рекурентною памʼяттю. КРОК 2: forward + вибір дії
// + anti-stuck + reward + накопичення ВІДРІЗКА у буфер послідовностей. Навчання
// (BPTT) ще НЕ підключене (крок 3) — ваги GRU поки не міняються, рій діє випадково;
// але дані для навчання вже течуть у seqReplay, а метрики reward/blind оживають.
func (b *Brain) stepGRU(cur [baseInputs]float32, hitWall bool) int {
	// [МЕТРИКИ ПАМʼЯТІ] Вбивця ВСЕВИДЮЩИЙ (flow-field глобальний), і слот 13 у
	// нього — не visible, а швидкість цілі. Читати його як видимість не можна.
	visible := b.flowNav || cur[inVisible] > 0.5

	// [МЕТРИКИ] Рахуємо ЩОКАДРУ, а не раз на рішення. Інакше при прорідженні
	// вибірка зменшилась би в gruSkip разів і перестала бути порівнянною зі
	// стек-шляхом — а порівнянність тут головне, бо саме нею ми і міряємо памʼять.
	if b.hasPrev && !b.flowNav && !b.prevVisible {
		b.mBlindN++
		if b.progress > 0 {
			b.mBlindClosed++
		}
	}

	// [НАГОРОДА] Теж ЩОКАДРУ — і накопичується в нагороду поточного кроку з
	// дискаунтом. Без цього при прорідженні (N−1)/N сигналу зникало б: rewardFor ще
	// й СПОЖИВАЄ бойові лічильники (dmgDealt/Taken/kills), тож пропустити його виклик
	// означало б і втратити шкоду, і залишити лічильники брудними на наступний кадр.
	if b.hasPrev {
		r := b.rewardFor(hitWall, b.whiskerOf(b.gruPrevX[:], b.prevAction))
		b.gruAcc += b.gruAccPow * r
		b.gruAccPow *= b.net.gamma
	}

	// Anti-stuck — та сама сітка безпеки, що й у стек-режимі (по вусах кадру).
	// Прогрес — ВЛАСНИЙ внесок агента в потрібному напрямку (пряма для рою,
	// коридор для вбивці), щоб anti-stuck не сварив за обхід стіни.
	madeProgress := b.hasPrev && b.progressToward() > stuckProgressEps
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

	// [ВАЖІЛЬ BPTT] Між рішеннями дія ПОВТОРЮЄТЬСЯ. Стан h не рухається — саме тому
	// вісім кроків розгортки накривають 8×gruSkip кадрів замість восьми.
	//
	// Anti-stuck тут навмисно НЕ перериває інтервал: лічильник накопичується вище й
	// спрацює на найближчому рішенні. Ціна — до gruSkip кадрів довбання стіни (при 20
	// це 0.167с). Перебивати інтервал було б гірше: кроки перестали б бути рівними за
	// часом, і множник γ^gruSkip у цілі став би неправдою.
	if b.gruTick > 0 {
		b.gruTick--
		b.gruPrevX = cur
		b.prevVisible = visible
		b.hasPrev = true
		return b.lastAction
	}
	b.gruTick = b.net.mem.gruSkip - 1

	// Закриваємо ПОПЕРЕДНІЙ крок: (кадр рішення, дія, СУМА нагород за інтервал).
	if b.gruHasDec {
		b.seqX[b.seqN] = b.gruDecX
		b.seqA[b.seqN] = b.gruDecA
		b.seqR[b.seqN] = b.gruAcc
		b.seqN++
		if b.seqN == seqTotal {
			// Відрізок повний (burn-in + навчальні) → у спільний буфер.
			b.net.rememberSeq(sequence{x: b.seqX, a: b.seqA, r: b.seqR, xEnd: cur})
			b.seqN = 0
		}
	}
	b.gruAcc, b.gruAccPow = 0, 1

	// Рекурентний forward: несемо власний стан b.h між РІШЕННЯМИ (не кадрами).
	q, hNew := b.net.forwardGRU(cur, b.h)
	b.h = hNew

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

	b.gruDecX, b.gruDecA, b.gruHasDec = cur, action, true

	b.gruPrevX = cur
	b.prevAction = action
	b.prevVisible = visible
	b.hasPrev = true
	b.lastAction = action
	b.lastQ = q // [ФОРМА] тіло витягнеться туди, куди мережа хоче
	return action
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
// [BURN-IN проти stored-state problem] replay стартує з h=0, але перші seqBurnIn
// кадрів — лише ПРОГРІВ: forward без loss/градієнтів, щоб h накопичив реальний
// контекст ПОТОЧНИМИ вагами. Loss і BPTT — лише на навчальному вікні
// [seqBurnIn..seqTotal). Так стан на момент обрахунку помилки близький до того,
// що агент реально має під час дії (а не до «щойно народженого» h=0).
func (n *Net) tdUpdateSeq(seq sequence) {
	gStep := n.mem.gammaStep(n.gamma) // [N-STEP] дискаунт на крок відрізка (γ^gruSkip)

	// --- Фаза 1: forward живої мережі з кешем (по ВСЬОМУ відрізку, вкл. burn-in) ---
	var hArr [seqTotal + 1][gruHidden]float32 // hArr[t] = h_{t-1}; hArr[0]=0
	var zc, rc, cc [seqTotal][gruHidden]float32
	for t := 0; t < seqTotal; t++ {
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

	// --- Фаза 2: target-ціль ---
	// Target-мережу теж проганяємо по ВСЬОМУ відрізку (щоб її h був прогрітий).
	// qTgt[s] = Q_tgt у стані s. Bootstrap кроку t бере наступний стан: t+1 (в межах)
	// або xEnd (останній). TD рахуємо ЛИШЕ на навчальному вікні [seqBurnIn..seqTotal).
	var hT [gruHidden]float32
	var qTgt [seqTotal][brainActions]float32
	for t := 0; t < seqTotal; t++ {
		qTgt[t], hT = n.forwardGRUTarget(seq.x[t], hT)
	}
	qEnd, _ := n.forwardGRUTarget(seq.xEnd, hT)

	// TD-помилка на кроках навчального вікна (semi-gradient: ціль — константа).
	var td [seqTotal]float32
	for t := seqBurnIn; t < seqTotal; t++ {
		var qNext [brainActions]float32
		if t < seqTotal-1 {
			qNext = qTgt[t+1]
		} else {
			qNext = qEnd
		}
		// [N-STEP] Дискаунт на КРОК, а не на кадр: крок накриває gruSkip кадрів, і
		// нагорода в seq.r уже є сумою Σγ^k·r по інтервалу. Якби тут лишився qGamma,
		// цінність майбутнього завищувалась би у стільки разів, скільки кадрів злито
		// в один крок. Рівноважна Q при цьому не змінюється — та сама сума, лише
		// перегрупована, тож qClip підбирати заново не треба.
		target := clamp(seq.r[t]+gStep*qNext[argmaxQ(qNext)], -n.clip, n.clip)

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

	// Гортаємо назад ЛИШЕ по навчальному вікні до seqBurnIn (у burn-in кадри
	// градієнт не пускаємо — вони суто для прогріву h). dhNext на межі відкидаємо.
	var dhNext [gruHidden]float32 // градієнт, що тече з майбутнього кроку в h_t
	for t := seqTotal - 1; t >= seqBurnIn; t-- {
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
	// gruLearnRate < qLearnRate: рекурентні кроки тримаємо спокійнішими (стабільність).
	cg := func(v float32) float32 { return clamp(v, -gruGradClip, gruGradClip) }
	for i := 0; i < gruHidden; i++ {
		for m := 0; m < baseInputs; m++ {
			n.Wz[i][m] += gruLearnRate * cg(dWz[i][m])
			n.Wr[i][m] += gruLearnRate * cg(dWr[i][m])
			n.Wh[i][m] += gruLearnRate * cg(dWh[i][m])
		}
		for j := 0; j < gruHidden; j++ {
			n.Uz[i][j] += gruLearnRate * cg(dUz[i][j])
			n.Ur[i][j] += gruLearnRate * cg(dUr[i][j])
			n.Uh[i][j] += gruLearnRate * cg(dUh[i][j])
		}
		n.Bz[i] += gruLearnRate * cg(dBz[i])
		n.Br[i] += gruLearnRate * cg(dBr[i])
		n.Bh[i] += gruLearnRate * cg(dBh[i])
	}
	for a := 0; a < brainActions; a++ {
		for k := 0; k < gruHidden; k++ {
			n.Wq[a][k] += gruLearnRate * cg(dWq[a][k])
		}
		n.Bq[a] += gruLearnRate * cg(dBq[a])
	}

	n.clipGRU()

	n.syncCounter++
	if n.syncCounter >= qTargetSync {
		n.syncTarget()
		n.syncCounter = 0
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
