package main

import (
	"math/rand"
	"os"
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
//
// ЗАПУСК: BOIDS_LADDER=1 go test -run TestLadder -v -timeout 90m
//
// За прапорцем навмисно: щаблі йдуть хвилинами (щабель 2 — понад 6), і в звичайній
// сюїті вони б зробили `go test ./...` непридатним для щоденної роботи.
// ==========================================================================

// ladderSkip — драбина довга, тож у звичайному прогоні її пропускаємо.
func ladderSkip(t *testing.T) {
	t.Helper()
	if os.Getenv("BOIDS_LADDER") == "" {
		t.Skip("довга драбина; запуск: BOIDS_LADDER=1 go test -run TestLadder -v -timeout 90m")
	}
}

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
	ladderSkip(t)
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
	ladderSkip(t)
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

// ladderRung3 — ЩАБЕЛЬ 3: умова не проста, а ДОБУТОК двох входів.
//
// Щаблі 1-2 брали «біт → конкретна дія» — просте табличне відображення на 2 випадки.
// У бою ж потрібне ВІДНОСНЕ рішення: «якщо стиль ВЕДУЧИЙ → розвернутись, якщо
// ДЗЕРКАЛЬНИЙ → продовжувати». Правильна дія залежить від ДРУГОГО входу — власного
// напрямку руху, — тож це вже 2 × 8 = 16 випадків, і мережа мусить вивчити добуток, а
// не пошук.
//
// Це остання неперевірена відмінність між драбиною й оракулом. Затримку й γ беремо ті,
// що щабель 2 показав робочими (65 кроків, γ = 0.99), щоб міряти РІВНО добуток.
func ladderRung3(delay int, gamma float32, steps int) (solved bool, hit float32) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)

	// state: [0]=стиль, [1..2]=власний напрямок руху (одиничний вектор), [3]=фаза,
	//        [4]=рішення вже ухвалене
	st := func(style, dir, phase, committed int) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(style)
		s[1], s[2] = dirs8[dir][0], dirs8[dir][1]
		s[3] = float32(phase) / float32(delay+1)
		s[4] = float32(committed)
		return s
	}
	// ВЕДУЧИЙ (0) → розвернутись: дія навпроти напрямку руху.
	// ДЗЕРКАЛЬНИЙ (1) → продовжувати: дія збігається з напрямком.
	want := func(style, dir int) int {
		if style == 0 {
			return (dir + brainActions/2) % brainActions
		}
		return dir
	}

	done, right, total := 0, 0, 0
	for done < steps {
		style, dir := rand.Intn(2), rand.Intn(brainActions)
		s0 := st(style, dir, 0, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want(style, dir) {
			committed = 1
		}
		if done > steps/2 {
			total++
			if committed == 1 {
				right++
			}
		}
		prev, prevA := s0, a0
		for ph := 1; ph <= delay; ph++ {
			cur := st(style, dir, ph, committed)
			n.remember(transition{s: prev, a: prevA, r: 0, s2: cur})
			n.train(qBatch)
			done++
			prev, prevA = cur, b.selectAction(cur)
		}
		n.remember(transition{s: prev, a: prevA, r: float32(committed), s2: prev, terminal: true})
		n.train(qBatch)
		done++
	}

	// Жадібно перевіряємо ВСІ 16 комбінацій.
	ok := 0
	for style := 0; style < 2; style++ {
		for dir := 0; dir < brainActions; dir++ {
			q, _, _ := n.forwardQ(st(style, dir, 0, 0))
			if argmaxQ(q) == want(style, dir) {
				ok++
			}
		}
	}
	if total > 0 {
		hit = 100 * float32(right) / float32(total)
	}
	return ok == 2*brainActions, hit
}

// TestLadderRung3 — чи бере учень ДОБУТОК двох входів, а не просто пошук по біту.
func TestLadderRung3(t *testing.T) {
	ladderSkip(t)
	const runs, steps = 4, 120000
	t.Logf("ЩАБЕЛЬ 3: дія = f(стиль, власний напрямок), 2×8 = 16 випадків")
	t.Logf("%8s %8s %12s %14s", "затримка", "γ", "усі 16", "точність")
	for _, c := range []struct {
		delay int
		gamma float32
	}{{0, 0.95}, {65, 0.99}} {
		ok, acc := 0, float32(0)
		for r := 0; r < runs; r++ {
			s, h := ladderRung3(c.delay, c.gamma, steps)
			if s {
				ok++
			}
			acc += h
		}
		t.Logf("%8d %8.2f %6d з %d %12.1f%%", c.delay, c.gamma, ok, runs, acc/float32(runs))
	}
}

// nStepBuf — накопичувач n-step переходів.
//
// Замість (s_t, a_t, r_t, s_{t+1}) віддає (s_t, a_t, Σγ^k·r_{t+k}, s_{t+n}). Кредит
// стрибає одразу на n кроків, а не повзе по одному за оновлення. Бутстрап при цьому
// мусить дисконтуватись на γ^n — це робить memContract.gammaStep, тож тут лише сума.
type nStepBuf struct {
	n     int
	gamma float32
	net   *Net
	s     [][brainInputs]float32
	a     []int
	r     []float32
}

// push — додати крок. Коли вікно набралось, віддає найстарший перехід у буфер мережі.
func (b *nStepBuf) push(s [brainInputs]float32, a int, r float32, next [brainInputs]float32) {
	b.s = append(b.s, s)
	b.a = append(b.a, a)
	b.r = append(b.r, r)
	if len(b.r) < b.n {
		return
	}
	b.net.remember(transition{s: b.s[0], a: b.a[0], r: b.sum(), s2: next})
	b.s, b.a, b.r = b.s[1:], b.a[1:], b.r[1:]
}

// flush — кінець епізоду: віддати ВСІ недороблені переходи як термінальні.
//
// Термінальні навмисно: далі нагород не буде, тож бутстрапити нема з чого. Без цього
// хвіст епізоду або губився б, або отримував завищену ціль.
func (b *nStepBuf) flush(last [brainInputs]float32) {
	for len(b.r) > 0 {
		b.net.remember(transition{s: b.s[0], a: b.a[0], r: b.sum(), s2: last, terminal: true})
		b.s, b.a, b.r = b.s[1:], b.a[1:], b.r[1:]
	}
}

func (b *nStepBuf) sum() float32 {
	var acc, pow float32 = 0, 1
	for _, v := range b.r {
		acc += pow * v
		pow *= b.gamma
	}
	return acc
}

// ladderRung3N — щабель 3 (добуток + затримка) з n-step.
func ladderRung3N(delay, nStep int, gamma float32, steps int) (solved bool, hit float32) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.mem.nStep = nStep
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)

	st := func(style, dir, phase, committed int) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(style)
		s[1], s[2] = dirs8[dir][0], dirs8[dir][1]
		s[3] = float32(phase) / float32(delay+1)
		s[4] = float32(committed)
		return s
	}
	want := func(style, dir int) int {
		if style == 0 {
			return (dir + brainActions/2) % brainActions
		}
		return dir
	}

	done, right, total := 0, 0, 0
	for done < steps {
		buf := &nStepBuf{n: nStep, gamma: n.gamma, net: n}
		style, dir := rand.Intn(2), rand.Intn(brainActions)
		s0 := st(style, dir, 0, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want(style, dir) {
			committed = 1
		}
		if done > steps/2 {
			total++
			if committed == 1 {
				right++
			}
		}
		prev, prevA := s0, a0
		for ph := 1; ph <= delay; ph++ {
			cur := st(style, dir, ph, committed)
			buf.push(prev, prevA, 0, cur)
			n.train(qBatch)
			done++
			prev, prevA = cur, b.selectAction(cur)
		}
		buf.push(prev, prevA, float32(committed), prev)
		buf.flush(prev)
		n.train(qBatch)
		done++
	}

	ok := 0
	for style := 0; style < 2; style++ {
		for dir := 0; dir < brainActions; dir++ {
			q, _, _ := n.forwardQ(st(style, dir, 0, 0))
			if argmaxQ(q) == want(style, dir) {
				ok++
			}
		}
	}
	if total > 0 {
		hit = 100 * float32(right) / float32(total)
	}
	return ok == 2*brainActions, hit
}

// TestLadderNStep — чи лікує n-step ту саму комірку, що падала 0 з 4.
func TestLadderNStep(t *testing.T) {
	ladderSkip(t)
	const runs, steps = 4, 120000
	t.Logf("ДОБУТОК 2×8 + затримка 65, γ0.99 — комірка, що падала 0 з 4")
	t.Logf("%8s %10s %14s", "n-step", "усі 16", "точність")
	best := 0
	for _, ns := range []int{1, 5, 20, 66} {
		ok, acc := 0, float32(0)
		for r := 0; r < runs; r++ {
			s, h := ladderRung3N(65, ns, 0.99, steps)
			if s {
				ok++
			}
			acc += h
		}
		if ok > best {
			best = ok
		}
		mark := ""
		if ns == 1 {
			mark = "  ← як зараз у грі"
		}
		if ns == 66 {
			mark = "  ← накриває всю затримку"
		}
		t.Logf("%8d %6d з %d %12.1f%%%s", ns, ok, runs, acc/float32(runs), mark)
	}
	if best == 0 {
		t.Error("n-step не лікує: затримка не є причиною, шукати далі")
	}
}

// ladderRung3Dense — та сама задача, але в буфер ідуть ЛИШЕ переходи РІШЕННЯ.
//
// n-step не полікував нічого, включно з n = 66, який несе нагороду прямо на стан
// рішення. Отже затримка не причина. Лишається розведення БУФЕРА:
//
//	епізод 66 кроків, рішення на ОДНОМУ → переходи рішення це 1.5% буфера,
//	решта 98.5% — заповнювачі з нульовою нагородою, а train семплить РІВНОМІРНО.
//
// Тут заповнювачі просто не зберігаємо. Якщо задача розвʼязується — причина знайдена, і
// вона не в кредиті й не в затримці, а в тому, ЩО ЛЕЖИТЬ У БУФЕРІ.
func ladderRung3Dense(delay int, gamma float32, steps int) (solved bool, hit float32) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.mem.nStep = delay + 1 // один перехід накриває весь епізод
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)

	st := func(style, dir, phase, committed int) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(style)
		s[1], s[2] = dirs8[dir][0], dirs8[dir][1]
		s[3] = float32(phase) / float32(delay+1)
		s[4] = float32(committed)
		return s
	}
	want := func(style, dir int) int {
		if style == 0 {
			return (dir + brainActions/2) % brainActions
		}
		return dir
	}

	done, right, total := 0, 0, 0
	for done < steps {
		style, dir := rand.Intn(2), rand.Intn(brainActions)
		s0 := st(style, dir, 0, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want(style, dir) {
			committed = 1
		}
		if done > steps/2 {
			total++
			if committed == 1 {
				right++
			}
		}
		// Проміжні кроки ВІДБУВАЮТЬСЯ (агент діє, час іде), але в буфер НЕ йдуть.
		//
		// ⚠️ train() кличемо ЩОКРОКУ, а не раз на епізод. Перша версія цього тесту
		// мала виклик у тому ж циклі, що й запис у буфер, — і щільна версія отримала
		// у 66 разів МЕНШЕ градієнтних кроків, ніж та, з якою її порівнювали. Тест
		// відпрацював за 1.8с замість 266с, і це було єдиною ознакою підміни.
		// Порівнювати треба РІВНИЙ бюджет навчання, інакше міряємо не склад буфера.
		for ph := 1; ph <= delay; ph++ {
			b.selectAction(st(style, dir, ph, committed))
			n.train(qBatch)
			done++
		}
		// Нагорода дисконтується вручну на весь епізод — так само, як це зробив би
		// n-step, тільки без заповнювачів у буфері.
		var pow float32 = 1
		for i := 0; i < delay; i++ {
			pow *= n.gamma
		}
		n.remember(transition{s: s0, a: a0, r: pow * float32(committed),
			s2: st(style, dir, delay, committed), terminal: true})
		n.train(qBatch)
		done++
	}

	ok := 0
	for style := 0; style < 2; style++ {
		for dir := 0; dir < brainActions; dir++ {
			q, _, _ := n.forwardQ(st(style, dir, 0, 0))
			if argmaxQ(q) == want(style, dir) {
				ok++
			}
		}
	}
	if total > 0 {
		hit = 100 * float32(right) / float32(total)
	}
	return ok == 2*brainActions, hit
}

// TestLadderDilution — чи причина в РОЗВЕДЕННІ буфера заповнювачами.
func TestLadderDilution(t *testing.T) {
	ladderSkip(t)
	const runs, steps = 4, 120000
	t.Logf("ДОБУТОК 2×8 + затримка 65, γ0.99 — та сама комірка")
	t.Logf("%34s %10s %12s", "буфер", "усі 16", "точність")
	okD, accD := 0, float32(0)
	for r := 0; r < runs; r++ {
		s, h := ladderRung3Dense(65, 0.99, steps)
		if s {
			okD++
		}
		accD += h
	}
	t.Logf("%34s %6d з %d %10.1f%%", "лише переходи РІШЕННЯ (100%)", okD, runs, accD/float32(runs))
	t.Logf("%34s %6d з %d %10.1f%%", "з заповнювачами (1.5%) — раніше", 0, runs, 13.4)
	if okD == 0 {
		t.Error("розведення буфера теж не причина — шукати далі")
	}
}

// trainStratified — семплити батч НЕ рівномірно: половину з переходів, де є нагорода.
//
// Найдешевша форма пріоритетного реплею. Повний PER семплить пропорційно |TD-помилці|
// й потребує дерева сум; тут ми користуємось тим, що в розрідженій задачі «є нагорода»
// і «велика TD-помилка» — майже одне й те саме, і ділимо буфер на дві купки.
//
// Питання, на яке відповідає: чи можна дістати ту саму якість, що дав чистий буфер
// (77.6% проти 13.4%), НЕ викидаючи заповнювачі — бо в грі їх не викинеш, кадри
// відбуваються.
func trainStratified(n *Net, k int, frac float32) {
	m := n.replayLen()
	if m < qMinReplay {
		return
	}
	var hot []int
	for i := 0; i < m; i++ {
		if r := n.replay[i].r; r > 0.001 || r < -0.001 {
			hot = append(hot, i)
		}
	}
	for i := 0; i < k; i++ {
		idx := rand.Intn(m)
		if len(hot) > 0 && rand.Float32() < frac {
			idx = hot[rand.Intn(len(hot))]
		}
		t := n.replay[idx]
		n.tdUpdate(t.s, t.a, t.r, t.s2, t.terminal)
	}
}

// ladderRung3Prio — задача з заповнювачами в буфері, але зі стратифікованим семплінгом.
func ladderRung3Prio(delay int, gamma, frac float32, steps int) (hit float32, allOK bool) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)

	st := func(style, dir, phase, committed int) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(style)
		s[1], s[2] = dirs8[dir][0], dirs8[dir][1]
		s[3] = float32(phase) / float32(delay+1)
		s[4] = float32(committed)
		return s
	}
	want := func(style, dir int) int {
		if style == 0 {
			return (dir + brainActions/2) % brainActions
		}
		return dir
	}

	done, right, total := 0, 0, 0
	for done < steps {
		style, dir := rand.Intn(2), rand.Intn(brainActions)
		s0 := st(style, dir, 0, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want(style, dir) {
			committed = 1
		}
		if done > steps/2 {
			total++
			if committed == 1 {
				right++
			}
		}
		prev, prevA := s0, a0
		for ph := 1; ph <= delay; ph++ {
			cur := st(style, dir, ph, committed)
			n.remember(transition{s: prev, a: prevA, r: 0, s2: cur})
			trainStratified(n, qBatch, frac)
			done++
			prev, prevA = cur, b.selectAction(cur)
		}
		n.remember(transition{s: prev, a: prevA, r: float32(committed), s2: prev, terminal: true})
		trainStratified(n, qBatch, frac)
		done++
	}

	ok := 0
	for style := 0; style < 2; style++ {
		for dir := 0; dir < brainActions; dir++ {
			q, _, _ := n.forwardQ(st(style, dir, 0, 0))
			if argmaxQ(q) == want(style, dir) {
				ok++
			}
		}
	}
	if total > 0 {
		hit = 100 * float32(right) / float32(total)
	}
	return hit, ok == 2*brainActions
}

// TestLadderPrioritized — чи відновлює стратифікований семплінг те, що дав чистий буфер.
func TestLadderPrioritized(t *testing.T) {
	ladderSkip(t)
	const runs, steps = 3, 120000
	t.Logf("ДОБУТОК 2×8 + затримка 65, γ0.99, заповнювачі в буфері ЛИШАЮТЬСЯ")
	t.Logf("%28s %12s %10s", "частка «гарячих» у батчі", "точність", "усі 16")
	for _, frac := range []float32{0.0, 0.25, 0.5, 0.9} {
		acc, ok := float32(0), 0
		for r := 0; r < runs; r++ {
			h, a := ladderRung3Prio(65, 0.99, frac, steps)
			acc += h
			if a {
				ok++
			}
		}
		mark := ""
		if frac == 0 {
			mark = "  ← рівномірно, як зараз"
		}
		t.Logf("%28.2f %10.1f%% %6d з %d%s", frac, acc/float32(runs), ok, runs, mark)
	}
	t.Logf("%28s %10.1f%%", "(чистий буфер, для порівняння)", 77.6)
}

// ladderRung3Grid — 2×2: ЯРЛИК × РОЗВЕДЕННЯ, усе інше однакове.
//
// Розділяє два фактори, які в попередніх тестах мінялись РАЗОМ:
//
//	ЯРЛИК       чи несуть проміжні стани committed («рішення вже правильне») — ознаку,
//	            що прямо каже відповідь і пояснює 98.5% буфера;
//	РОЗВЕДЕННЯ  чи лежать у буфері заповнювачі (65 з 66 переходів) чи лише рішення.
//
// Кредит в ОБОХ випадках доносить n-step на весь епізод, тож затримка тут не фактор —
// її вже виключено (n = 66 не змінив нічого).
//
// Бюджет навчання РІВНИЙ: train() щокроку в усіх комірках. Перша версія тесту на
// розведення цього не мала — щільна комірка отримала у 66 разів менше градієнтних
// кроків, і я мало не оголосив розведення непричетним.
//
// Комірка «без ярлика, без розведення» — контроль: якщо задача не розвʼязується навіть
// там, значить ярлик робив УСЮ роботу, і 77.6% були ілюзією.
func ladderRung3Grid(shortcut, dense bool, delay int, gamma float32, steps int) (hit float32, allOK bool) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.mem.nStep = delay + 1
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)

	st := func(style, dir, phase, committed int) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(style)
		s[1], s[2] = dirs8[dir][0], dirs8[dir][1]
		s[3] = float32(phase) / float32(delay+1)
		if shortcut {
			s[4] = float32(committed)
		}
		return s
	}
	want := func(style, dir int) int {
		if style == 0 {
			return (dir + brainActions/2) % brainActions
		}
		return dir
	}
	// Дисконт нагороди на всю затримку — однаковий в обох гілках.
	var pow float32 = 1
	for i := 0; i < delay; i++ {
		pow *= n.gamma
	}

	done, right, total := 0, 0, 0
	for done < steps {
		style, dir := rand.Intn(2), rand.Intn(brainActions)
		s0 := st(style, dir, 0, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want(style, dir) {
			committed = 1
		}
		if done > steps/2 {
			total++
			if committed == 1 {
				right++
			}
		}
		var buf *nStepBuf
		if !dense {
			buf = &nStepBuf{n: delay + 1, gamma: n.gamma, net: n}
		}
		prev, prevA := s0, a0
		for ph := 1; ph <= delay; ph++ {
			cur := st(style, dir, ph, committed)
			if !dense {
				buf.push(prev, prevA, 0, cur)
			}
			n.train(qBatch) // рівний бюджет в ОБОХ гілках
			done++
			prev, prevA = cur, b.selectAction(cur)
		}
		if dense {
			// Лише перехід РІШЕННЯ, з тією самою дисконтованою нагородою.
			n.remember(transition{s: s0, a: a0, r: pow * float32(committed),
				s2: prev, terminal: true})
		} else {
			buf.push(prev, prevA, float32(committed), prev)
			buf.flush(prev)
		}
		n.train(qBatch)
		done++
	}

	ok := 0
	for style := 0; style < 2; style++ {
		for dir := 0; dir < brainActions; dir++ {
			q, _, _ := n.forwardQ(st(style, dir, 0, 0))
			if argmaxQ(q) == want(style, dir) {
				ok++
			}
		}
	}
	if total > 0 {
		hit = 100 * float32(right) / float32(total)
	}
	return hit, ok == 2*brainActions
}

// TestLadderGrid — ЯРЛИК × РОЗВЕДЕННЯ, повний 2×2.
func TestLadderGrid(t *testing.T) {
	ladderSkip(t)
	const runs, steps = 4, 120000
	t.Logf("ДОБУТОК 2×8 + затримка 65, γ0.99, кредит доносить n-step у ВСІХ комірках")
	t.Logf("%10s %14s %12s %10s", "ярлик", "буфер", "точність", "усі 16")
	for _, sc := range []bool{true, false} {
		for _, dense := range []bool{false, true} {
			acc, ok := float32(0), 0
			for r := 0; r < runs; r++ {
				h, a := ladderRung3Grid(sc, dense, 65, 0.99, steps)
				acc += h
				if a {
					ok++
				}
			}
			lblS, lblB := "є", "розведений"
			if !sc {
				lblS = "НЕМАЄ"
			}
			if dense {
				lblB = "лише рішення"
			}
			t.Logf("%10s %14s %10.1f%% %6d з %d", lblS, lblB, acc/float32(runs), ok, runs)
		}
	}
	t.Logf("випадковий рівень 12.5%%")
}

// trainPER — пріоритетний реплей за |TD-помилкою|. Справжня версія, а не сурогат.
//
// Попередня спроба (trainStratified) семплила за НАГОРОДОЮ і дала нуль. Причина не в
// ідеї, а в ознаці: у грі кожен кадр є рішенням, поділу на «рішення» й «заповнювач»
// немає, і нагорода зовсім не там, де важлива дія. Єдине, що це виражає, — наскільки
// перехід ще ДИВУЄ мережу, тобто |TD|.
//
// Спрощення проти канонічного PER: без дерева сум і без importance-sampling ваг.
// Пріоритет оновлюється на кожному навчанні переходу, нові переходи входять із
// максимальним (щоб їх узагалі побачили хоч раз). alpha = 0 дає рівномірний семплінг —
// це наша базова лінія в тій самій функції, без окремого шляху коду.
type perBuf struct {
	prio []float32
}

func (p *perBuf) train(n *Net, k int, alpha float32) {
	m := n.replayLen()
	if m < qMinReplay {
		return
	}
	for len(p.prio) < m {
		p.prio = append(p.prio, 1e9) // новий перехід — максимальний пріоритет
	}
	// Сума ваг (prio^alpha). При alpha = 0 усі ваги 1 → рівномірно.
	var total float32
	w := make([]float32, m)
	for i := 0; i < m; i++ {
		v := p.prio[i]
		if v > 1e8 {
			v = 10 // стеля для «ще не бачених», щоб вони не з'їдали весь батч
		}
		if alpha == 0 {
			w[i] = 1
		} else {
			w[i] = v + 0.01 // епсілон, щоб нульові помилки все ще мали шанс
		}
		total += w[i]
	}
	for s := 0; s < k; s++ {
		r, idx := rand.Float32()*total, m-1
		var acc float32
		for i := 0; i < m; i++ {
			acc += w[i]
			if acc >= r {
				idx = i
				break
			}
		}
		t := n.replay[idx]
		td := n.tdUpdate(t.s, t.a, t.r, t.s2, t.terminal)
		if td < 0 {
			td = -td
		}
		p.prio[idx] = td
	}
}

// ladderRung3PER — розведений буфер, БЕЗ ярлика, з пріоритетним реплеєм.
//
// Це рівно та комірка, що дала 41.7% при рівномірному семплінгу, тоді як чистий буфер
// дає 78.9%. Питання: чи закриває PER цю прірву, не викидаючи заповнювачі — бо в грі
// їх не викинеш, кадри відбуваються.
func ladderRung3PER(alpha float32, delay int, gamma float32, steps int) (hit float32, allOK bool) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.mem.nStep = delay + 1
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)
	per := &perBuf{}

	st := func(style, dir, phase int) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(style)
		s[1], s[2] = dirs8[dir][0], dirs8[dir][1]
		s[3] = float32(phase) / float32(delay+1)
		return s // ярлика НЕМАЄ навмисно
	}
	want := func(style, dir int) int {
		if style == 0 {
			return (dir + brainActions/2) % brainActions
		}
		return dir
	}

	done, right, total := 0, 0, 0
	for done < steps {
		style, dir := rand.Intn(2), rand.Intn(brainActions)
		s0 := st(style, dir, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want(style, dir) {
			committed = 1
		}
		if done > steps/2 {
			total++
			if committed == 1 {
				right++
			}
		}
		buf := &nStepBuf{n: delay + 1, gamma: n.gamma, net: n}
		prev, prevA := s0, a0
		for ph := 1; ph <= delay; ph++ {
			cur := st(style, dir, ph)
			buf.push(prev, prevA, 0, cur)
			per.train(n, qBatch, alpha)
			done++
			prev, prevA = cur, b.selectAction(cur)
		}
		buf.push(prev, prevA, float32(committed), prev)
		buf.flush(prev)
		per.train(n, qBatch, alpha)
		done++
	}

	ok := 0
	for style := 0; style < 2; style++ {
		for dir := 0; dir < brainActions; dir++ {
			q, _, _ := n.forwardQ(st(style, dir, 0))
			if argmaxQ(q) == want(style, dir) {
				ok++
			}
		}
	}
	if total > 0 {
		hit = 100 * float32(right) / float32(total)
	}
	return hit, ok == 2*brainActions
}

// TestLadderPER — чи закриває пріоритетний реплей прірву 41.7% → 78.9%.
func TestLadderPER(t *testing.T) {
	ladderSkip(t)
	const runs, steps = 3, 60000
	t.Logf("Розведений буфер, БЕЗ ярлика, добуток + затримка 65, γ0.99")
	t.Logf("%10s %12s %10s", "alpha", "точність", "усі 16")
	for _, a := range []float32{0, 1} {
		acc, ok := float32(0), 0
		for r := 0; r < runs; r++ {
			h, all := ladderRung3PER(a, 65, 0.99, steps)
			acc += h
			if all {
				ok++
			}
		}
		lbl := "рівномірно"
		if a > 0 {
			lbl = "за |TD|"
		}
		t.Logf("%10s %10.1f%% %6d з %d", lbl, acc/float32(runs), ok, runs)
	}
	t.Logf("орієнтири: чистий буфер 78.9%%, випадковий рівень 12.5%%")
}

// ladderFaithful — драбина з ЧЕСНИМИ проміжними станами.
//
// [ЩО БУЛО НЕ ТАК] У попередніх щаблях заповнювачі відрізнялись від стану рішення лише
// слотом фази: s0 мав phase = 0, s1 — 1/66 = 0.015. Для гладкого апроксиматора це майже
// той самий вхід. А цілі в заповнювачах ~нуль (стан не кодує гілку, тож ±0.52
// усереднюються), і 65 із 66 навчальних кроків тягнули до нуля область, яка ЗБІГАЄТЬСЯ
// зі станом рішення. Сигнал розмивався через УЗАГАЛЬНЕННЯ, а не через рідкість.
//
// Саме тому нічого не давали ні n-step, ні пріоритетний реплей: обидва бʼють у частоту
// й кредит, а проблема була в ІНТЕРФЕРЕНЦІЇ. І саме тому висновок «розведення буфера —
// причина» довелось відкликати вдруге: обидва тести, що його «підтверджували», ділили
// цю ваду.
//
// [ЯК ЧЕСНО] У грі за 65 кадрів позиція юніта змінюється СИЛЬНО, і стани не бувають
// майже тотожними. Тут так само: агент рухається згідно з обраними діями, і зміщення
// входить у стан. Це НЕ ярлик — воно каже, ЩО агент зробив, а не ЧИ це було правильно
// (саме цим грішив committed).
func ladderFaithful(delay int, gamma float32, steps int, dense bool) (hit float32, allOK bool) {
	saved := qEpsilonConst
	qEpsilonConst = 0.05
	defer func() { qEpsilonConst = saved }()

	n := NewNet()
	n.mem = resolveMemContract(MemoryStack, 1, 10, 0, 0)
	n.mem.nStep = delay + 1
	n.gamma, n.clip = resolveHorizon(gamma, 0)
	b := NewBrainWith(n)

	// state: [0]=стиль, [1..2]=напрямок «руху цілі», [3]=фаза, [4..5]=ВЛАСНЕ зміщення
	st := func(style, dir, phase int, ox, oy float32) [brainInputs]float32 {
		var s [brainInputs]float32
		s[0] = float32(style)
		s[1], s[2] = dirs8[dir][0], dirs8[dir][1]
		s[3] = float32(phase) / float32(delay+1)
		s[4], s[5] = ox, oy
		return s
	}
	want := func(style, dir int) int {
		if style == 0 {
			return (dir + brainActions/2) % brainActions
		}
		return dir
	}

	done, right, total := 0, 0, 0
	for done < steps {
		style, dir := rand.Intn(2), rand.Intn(brainActions)
		s0 := st(style, dir, 0, 0, 0)
		a0 := b.selectAction(s0)
		committed := -1
		if a0 == want(style, dir) {
			committed = 1
		}
		if done > steps/2 {
			total++
			if committed == 1 {
				right++
			}
		}
		var buf *nStepBuf
		if !dense {
			buf = &nStepBuf{n: delay + 1, gamma: n.gamma, net: n}
		}
		// Зміщення накопичується з РЕАЛЬНО обраних дій — як позиція юніта в грі.
		ox, oy := dirs8[a0][0], dirs8[a0][1]
		prev, prevA := s0, a0
		for ph := 1; ph <= delay; ph++ {
			cur := st(style, dir, ph, ox/float32(delay), oy/float32(delay))
			if !dense {
				buf.push(prev, prevA, 0, cur)
			}
			n.train(qBatch)
			done++
			prevA = b.selectAction(cur)
			ox += dirs8[prevA][0]
			oy += dirs8[prevA][1]
			prev = cur
		}
		if dense {
			var pow float32 = 1
			for i := 0; i < delay; i++ {
				pow *= n.gamma
			}
			n.remember(transition{s: s0, a: a0, r: pow * float32(committed),
				s2: prev, terminal: true})
		} else {
			buf.push(prev, prevA, float32(committed), prev)
			buf.flush(prev)
		}
		n.train(qBatch)
		done++
	}

	ok := 0
	for style := 0; style < 2; style++ {
		for dir := 0; dir < brainActions; dir++ {
			q, _, _ := n.forwardQ(st(style, dir, 0, 0, 0))
			if argmaxQ(q) == want(style, dir) {
				ok++
			}
		}
	}
	if total > 0 {
		hit = 100 * float32(right) / float32(total)
	}
	return hit, ok == 2*brainActions
}

// TestLadderFaithful — чи лишається провал, коли заповнювачі перестають бути
// майже-копіями стану рішення.
func TestLadderFaithful(t *testing.T) {
	ladderSkip(t)
	const runs, steps = 4, 120000
	t.Logf("ЧЕСНІ заповнювачі: стан несе ВЛАСНЕ зміщення від обраних дій")
	t.Logf("%16s %12s %10s", "буфер", "точність", "усі 16")
	for _, dense := range []bool{false, true} {
		acc, ok := float32(0), 0
		for r := 0; r < runs; r++ {
			h, a := ladderFaithful(65, 0.99, steps, dense)
			acc += h
			if a {
				ok++
			}
		}
		lbl := "розведений"
		if dense {
			lbl = "лише рішення"
		}
		t.Logf("%16s %10.1f%% %6d з %d", lbl, acc/float32(runs), ok, runs)
	}
	t.Logf("для порівняння зі СТАРОЮ (нечесною) конструкцією: 39.3%% / 78.9%%")
}
