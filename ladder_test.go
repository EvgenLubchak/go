package main

import (
	"math/rand"
	"testing"
)

// ==========================================================================
// ДРАБИНА: мінімальна задача, де ми контролюємо ПО ОДНОМУ фактору за раз.
//
// НАВІЩО. Стенд чесно міряє ГРУ, але коли виходить нуль, він не каже, ДЕ зламалось —
// у механіці, нагороді, кредиті, ємності чи памʼяті. За один день ми перебрали ці
// варіанти по одному, кожен по годині прогону, і щоразу дізнавались причину ПІСЛЯ.
//
// Тут немає ні фізики, ні стін, ні бою, ні сусідів — лише мережа, ε-greedy, буфер і
// tdUpdate, тобто ТОЙ САМИЙ учень, що в грі. Тому результат переноситься.
//
// Щаблі (кожен додає рівно один фактор):
//
//	1  нагорода ОДРАЗУ, умова видима   → ємність і дослідження
//	2  нагорода із затримкою            → поширення кредиту
//	3  нагорода розріджена              → кількість сигналу
//	4  умову видно лише на початку      → памʼять
//
// Зламалось на щаблі N — далі не йдемо: наступні щаблі про причини, яких ми ще не
// дійшли. Саме цього приладу нам бракувало весь час.
// ==========================================================================

// ladderState — вхід мережі для щабля 1: умова в слоті 0, решта нулі.
// Один слот навмисно: рівно так само в грі подавався біт стилю.
func ladderState(bit int) [brainInputs]float32 {
	var s [brainInputs]float32
	s[0] = float32(bit)
	return s
}

// ladderRung1 — чи здатен цей учень вивчити УМОВНУ політику взагалі.
//
// Задача мінімальна: «біт 0 → дія 0, біт 1 → дія 4». Нагорода ±1 ОДРАЗУ, умова видима
// щокадру, памʼять не потрібна, кредит не треба нікуди тягнути. Якщо не вивчить ЦЕ —
// усі бойові нулі пояснюються, і памʼять тут ні до чого.
//
// Повертає, чи вивчена ПОЛІТИКА правильна для обох умов (жадібно, без ε).
func ladderRung1(eps float32, steps int) (solved bool, acc float32) {
	saved := qEpsilonConst
	qEpsilonConst = eps
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0) // памʼять вимкнена: вона тут зайва
	b := NewBrainWith(n)
	want := [2]int{0, 4} // протилежні напрямки у dirs8

	var right, total int
	bit := rand.Intn(2)
	for i := 0; i < steps; i++ {
		s := ladderState(bit)
		a := b.selectAction(s)
		r := float32(-1)
		if a == want[bit] {
			r = 1
		}
		if i >= steps/2 { // точність рахуємо на другій половині — після розігріву
			total++
			if a == want[bit] {
				right++
			}
		}
		next := rand.Intn(2)
		n.remember(transition{s: s, a: a, r: r, s2: ladderState(next)})
		n.train(qBatch)
		bit = next
	}

	// Жадібна перевірка вивченої політики: без ε, просто argmax Q.
	solved = true
	for bt := 0; bt < 2; bt++ {
		q, _, _ := n.forwardQ(ladderState(bt))
		if argmaxQ(q) != want[bt] {
			solved = false
		}
	}
	if total > 0 {
		acc = 100 * float32(right) / float32(total)
	}
	return solved, acc
}

// TestLadderRung1 — ЩАБЕЛЬ 1 плюс свіп ε.
//
// ε тут не налаштування, а підозрюваний: щоб вивчити умовну політику, треба спробувати
// ОБИДВІ дії в ОБОХ умовах, а при ε = 0.01 агент відхиляється раз на сто кроків.
func TestLadderRung1(t *testing.T) {
	const runs, steps = 8, 20000
	t.Logf("ЩАБЕЛЬ 1: «біт 0 → дія 0, біт 1 → дія 4», нагорода ±1 одразу, памʼять вимкнена")
	t.Logf("%8s %10s %14s", "ε", "розвʼязано", "точність")
	best := 0
	for _, eps := range []float32{0.01, 0.05, 0.15, 0.30} {
		ok, accSum := 0, float32(0)
		for r := 0; r < runs; r++ {
			s, a := ladderRung1(eps, steps)
			if s {
				ok++
			}
			accSum += a
		}
		if ok > best {
			best = ok
		}
		mark := ""
		if eps == 0.01 {
			mark = "  ← як у грі"
		}
		t.Logf("%8.2f %6d з %d %12.1f%%%s", eps, ok, runs, accSum/float32(runs), mark)
	}
	if best == 0 {
		t.Errorf("учень НЕ вивчив найпростішу умовну політику за жодного ε — " +
			"усі бойові нулі пояснюються цим, і памʼять тут ні до чого")
	}
}

// ladderRung2 — ЩАБЕЛЬ 2: та сама умовна задача, але нагорода приходить ЧЕРЕЗ delay
// кроків після рішення.
//
// Це головна відмінність гри від щабля 1. У бою агент вирішує «почати ухилятись» на
// початку замаху, а наслідок (є шкода чи немає) настає через dashWindup + dashActive =
// 65 кадрів. При γ = 0.95 це 0.95^65 = 3.6% кредиту.
//
// Проміжні стани несуть «рішення вже ухвалене» — так само, як у грі його несе позиція
// юніта, що вже відійшов убік. Без цього цінність не мала б за що чіплятись.
func ladderRung2(delay int, gamma float32, steps int) (solved bool) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05 // трохи вище за грове: щабель 1 показав, що це не шкодить
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)
	want := [2]int{0, 4}

	// state: [0]=біт, [1]=фаза 0..1, [2]=чи правильно обрано (0 = ще не обрано)
	st := func(bit, phase, committed int) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(bit)
		s[1] = float32(phase) / float32(delay+1)
		s[2] = float32(committed)
		return s
	}

	done := 0
	for done < steps {
		bit := rand.Intn(2)
		s0 := st(bit, 0, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want[bit] {
			committed = 1
		}
		prev, prevA := s0, a0
		for ph := 1; ph <= delay; ph++ {
			cur := st(bit, ph, committed)
			n.remember(transition{s: prev, a: prevA, r: 0, s2: cur})
			n.train(qBatch)
			done++
			prev, prevA = cur, b.selectAction(cur)
		}
		// Термінальний крок: нагорода за рішення, ухвалене delay кроків тому.
		n.remember(transition{s: prev, a: prevA, r: float32(committed), s2: prev, terminal: true})
		n.train(qBatch)
		done++
	}

	solved = true
	for bt := 0; bt < 2; bt++ {
		q, _, _ := n.forwardQ(st(bt, 0, 0))
		if argmaxQ(q) != want[bt] {
			solved = false
		}
	}
	return solved
}

// TestLadderRung2 — чи доходить кредит крізь затримку, і чи рятує γ.
func TestLadderRung2(t *testing.T) {
	const runs, steps = 6, 60000
	t.Logf("ЩАБЕЛЬ 2: рішення на кроці 0, нагорода через delay кроків")
	t.Logf("%8s %10s %12s %12s", "затримка", "γ", "γ^затримка", "розвʼязано")
	any := false
	for _, delay := range []int{0, 15, 30, 65} {
		for _, gamma := range []float32{0.95, 0.99} {
			ok := 0
			for r := 0; r < runs; r++ {
				if ladderRung2(delay, gamma, steps) {
					ok++
				}
			}
			credit := float32(1)
			for i := 0; i < delay; i++ {
				credit *= gamma
			}
			mark := ""
			if delay == 65 && gamma == 0.95 {
				mark = "  ← як у бою"
			}
			if ok > 0 {
				any = true
			}
			t.Logf("%8d %10.2f %11.3f %6d з %d%s", delay, gamma, credit, ok, runs, mark)
		}
	}
	if !any {
		t.Error("кредит не доходить НІ ЗА ЯКОЇ затримки — проблема в поширенні цінності")
	}
}
