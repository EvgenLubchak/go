package main

import "math"

// ==========================================================================
// ШЛЯХ 1: FRAME-STACKING (памʼять «руками») — активний при useGRU = false.
//
// На вхід мережі йде СТЕК останніх stackFrames кадрів (семпли кожні stackSkip),
// тобто вікно історії фіксованої довжини, яке задали МИ. Мережа звичайна
// feedforward: brainInputs(64) → hidden1 → hidden2 → Q(8).
//
// Це історично перший підхід до памʼяті в цьому стенді. Він лишається як:
//   • BASELINE для порівняння (blind-chase ~55% проти ~61% у GRU),
//   • FALLBACK, якщо рекурентна політика попливе,
//   • предмет тестів ядра Q-learning (TestQLearningTDUpdate / ChasesNoWalls).
//
// Рекурентний шлях — у brain_gru.go. Спільне ядро — у brain.go.
// ==========================================================================

// transition — один крок досвіду: (стан, дія, нагорода, наступний стан).
// Це «одиниця пам'яті» для experience replay (frame-stacking шлях).
type transition struct {
	s  [brainInputs]float32
	a  int
	r  float32
	s2 [brainInputs]float32
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

// stepStack — крок агента у СТЕК-режимі (викликається з Brain.Step, коли
// useGRU=false). НЕ тренує мережу — лише кладе досвід у буфер; навчання йде
// раз/кадр однопотоково у g.trainBrains() (мережа може бути спільною).
func (b *Brain) stepStack(cur [baseInputs]float32, dist float32, hitWall bool) int {
	// [ПАМ'ЯТЬ] Склеюємо поточний кадр + історію → повний вхід мережі.
	// Поточний кадр — ПЕРШИЙ у стеку, тож whiskers лишаються на індексах 5..12
	// (тому maxWhisker/escapeAction/proximity-reward працюють без змін).
	stacked := b.buildStacked(cur)

	// [МЕТРИКИ ПАМʼЯТІ] Чи бачить агент гравця ЦЬОГО кадру (вхід visible = cur[13]).
	visible := cur[inVisible] > 0.5

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
		reward += rewardNearWall * b.prevState[inWhisker0+b.prevAction]

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
