package main

import (
	"fmt"
	"math"
	"testing"
)

// ==========================================================================
// ЧИСЛОВА ПЕРЕВІРКА ГРАДІЄНТА BPTT (gradient check).
//
// Навіщо: GRU у нас нерозрізненний від ВІДСУТНОСТІ памʼяті на 96 прогонах
// (p = 0.45), при тому що стек на тій самій задачі витягує реальні +2 пункти.
// Усі гіперпараметри перебрані — темп навчання (2 значення) і важіль BPTT
// (4 значення, від 0.067с до 2.67с). Лишилась гіпотеза «непрацюючий градієнт»,
// а поведінкові заміри її не розрізняють. Ось остаточна перевірка.
//
// TestGRULearnsSequence цього не ловить: він дивиться лише на ЗНАК сумарного
// оновлення на одному відрізку. Пройшов би й з неправильним градієнтом
// reset-ворота — прямий шлях Wh·x + Wq·h сам дав би те зростання.
//
// ТРИ ПАСТКИ, через які наївна перевірка дала б ФАЛЬШИВИЙ баг:
//
//  1. BPTT навмисно ОБРІЗАНА на seqBurnIn (dhNext на межі відкидається). Тобто
//     аналітичний градієнт не містить шляху в burn-in кадри. Тому числову похідну
//     теж рахуємо з ФІКСОВАНОГО h на вході вікна.
//  2. dhPrev кліпається gruGradClip — свідоме відхилення від справжнього
//     градієнта. На час перевірки кліп вимикаємо.
//  3. SEMI-GRADIENT: td — константа (ціль не диференціюємо). Отже
//     диференціювати треба J = Σ td_t·Q_t(a_t), а не ½err².
//
// ЯК ПОЗБУЛИСЬ РИЗИКУ ВЛАСНОЇ ПОМИЛКИ. Замість переписувати обчислення цілі
// (де легко розійтись із продакшеном і полювати на привида) робимо так, щоб
// td_t гарантовано САТУРУВАВСЯ до ±1: даємо нагороду ±50 при qClip=25. Тоді
// J = ±Σ Q_t(a_t), і єдине, що ми реалізуємо самі, — сума. Форвард беремо
// продакшеновий (forwardGRU), тож звірятись є з чим.
// ==========================================================================

// gradCheckEps — крок скінченної різниці. Усе в float32 (~7 значущих цифр), тож
// замалий крок дає катастрофічне скорочення, а завеликий — похибку обрізання.
// 1e-2 при вагах ~0.2 і гладких tanh/sigmoid лишає обидві похибки на рівні ~1e-4.
const gradCheckEps = 1e-2

// gradCheckTol — допустима ВІДНОСНА розбіжність. Ми шукаємо структурні помилки
// (відсутній член, переплутаний знак, не той множник) — вони дають розбіжність у
// рази, а не у відсотки. 5% ловить їх усі й не дає фальшивих тривог від float32.
const gradCheckTol = 0.05

// gradCheckFloor — градієнти дрібніші за це пропускаємо як нецікаві.
const gradCheckFloor = 1e-5

// gradCheckAbs — АБСОЛЮТНИЙ допуск. Без нього перевірка бракує саму себе: на
// градієнтах порядку 1e-4 скінченні різниці у float32 дають похибку кількох
// мільйонних, і ВІДНОСНА розбіжність виходить 5–7% при абсолютній 8e-6. Перший
// прогін цієї перевірки саме так і «знайшов баг» у reset-воротах, якого немає.
//
// Комбіноване правило |num−ana| ≤ abs + tol·max(|num|,|ana|) — стандарт для
// gradient check саме через це. Структурні помилки (відсутній член, знак, не той
// множник) дають розбіжність у РАЗИ, тож 2e-5 їх не приховає.
const gradCheckAbs = 2e-5

func TestGRUGradientMatchesFiniteDifference(t *testing.T) {
	savedClip, savedLR, savedGRU := gruGradClip, gruLearnRate, useGRU
	defer func() { gruGradClip, gruLearnRate, useGRU = savedClip, savedLR, savedGRU }()

	// Кліп ГЕТЬ: він навмисно спотворює градієнт, і з ним перевірка безглузда.
	gruGradClip = 1e9
	const lr = 1e-3
	gruLearnRate = lr
	useGRU = true

	for _, sign := range []float32{+1, -1} {
		name := "td=+1"
		if sign < 0 {
			name = "td=-1"
		}
		t.Run(name, func(t *testing.T) {
			runGradCheck(t, sign, lr)
		})
	}
}

func runGradCheck(t *testing.T, sign float32, lr float32) {
	n := NewNet()
	n.mem = resolveMemContract(MemoryGRU, 0, 0, 1, 0)
	n.syncTarget() // target = живі ваги (нам байдуже, ціль однаково сатурує)

	// Відрізок із детермінованими входами. Нагорода ±50 при qClip=25 гарантує, що
	// rawTD виходить за ±1 і td_t сатурує рівно до ±1 — далі перевіримо це явно.
	var seq sequence
	for i := 0; i < seqTotal; i++ {
		for m := 0; m < baseInputs; m++ {
			seq.x[i][m] = float32(math.Sin(float64(i*3+m*7))) * 0.5
		}
		seq.a[i] = (i * 3) % brainActions
		seq.r[i] = sign * 50
	}
	for m := 0; m < baseInputs; m++ {
		seq.xEnd[m] = float32(math.Cos(float64(m))) * 0.3
	}

	// --- Аналітичний градієнт: знімаємо з різниці ваг ---
	base := snapshotGRU(n)
	n.tdUpdateSeq(seq)
	after := snapshotGRU(n)
	restoreGRU(n, base)

	// Стан на вході навчального вікна — рахуємо БАЗОВИМИ вагами й тримаємо
	// фіксованим: саме так поводиться обрізана BPTT.
	var h0 [gruHidden]float32
	for i := 0; i < seqBurnIn; i++ {
		_, h0 = n.forwardGRU(seq.x[i], h0)
	}

	// Перевіряємо припущення про сатурацію td, а не віримо на слово.
	{
		h := h0
		for i := seqBurnIn; i < seqTotal; i++ {
			var q [brainActions]float32
			q, h = n.forwardGRU(seq.x[i], h)
			if qa := q[seq.a[i]]; math.Abs(float64(qa)) >= 1 {
				t.Fatalf("крок %d: |Q(a)| = %.3f ≥ 1 — td міг НЕ сатурувати, і вся перевірка недійсна",
					i, qa)
			}
		}
	}

	// J(w) = Σ td_t·Q_t(a_t) від ФІКСОВАНОГО h0. td_t = sign (сатуровано).
	objective := func() float64 {
		h := h0
		var s float64
		for i := seqBurnIn; i < seqTotal; i++ {
			var q [brainActions]float32
			q, h = n.forwardGRU(seq.x[i], h)
			s += float64(sign) * float64(q[seq.a[i]])
		}
		return s
	}

	// --- Порівняння по ГРУПАХ ваг: баг живе у формулі групи ---
	type group struct {
		name string
		live []*float32
		anal []float32
	}
	groups := collectGRUGroups(n, base, after, lr)

	checked, skipped := 0, 0
	for _, g := range groups {
		var worst float64
		var worstA, worstN float64
		for k := range g.live {
			analytic := float64(g.anal[k])
			if math.Abs(analytic) < gradCheckFloor {
				skipped++
				continue
			}
			p := g.live[k]
			orig := *p
			*p = orig + gradCheckEps
			jp := objective()
			*p = orig - gradCheckEps
			jm := objective()
			*p = orig

			numeric := (jp - jm) / (2 * gradCheckEps)
			scale := math.Max(math.Abs(analytic), math.Abs(numeric))
			absErr := math.Abs(numeric - analytic)
			checked++
			// Скільки допуску «зʼїдено»: 1.0 = рівно на межі.
			used := absErr / (gradCheckAbs + gradCheckTol*scale)
			if used > worst {
				worst, worstA, worstN = used, analytic, numeric
			}
		}
		if worst > 1 {
			t.Errorf("%s: допуск перевищено в %.2f× — аналітично %.6g, численно %.6g (розбіжність %.2g)",
				g.name, worst, worstA, worstN, math.Abs(worstN-worstA))
		} else {
			t.Logf("%s: використано %.0f%% допуску (найгірше: %.6g проти %.6g)",
				g.name, 100*worst, worstA, worstN)
		}
	}

	if checked < 40 {
		t.Fatalf("перевірено лише %d ваг (пропущено %d як надто дрібні) — тест вийшов пустим",
			checked, skipped)
	}
	t.Logf("перевірено %d ваг, пропущено %d (|градієнт| < %g)", checked, skipped, gradCheckFloor)
}

// --- допоміжне: знімок і відновлення ваг GRU ---

type gruSnapshot struct {
	Wz, Wr, Wh [gruHidden][baseInputs]float32
	Uz, Ur, Uh [gruHidden][gruHidden]float32
	Bz, Br, Bh [gruHidden]float32
	Wq         [brainActions][gruHidden]float32
	Bq         [brainActions]float32
}

func snapshotGRU(n *Net) gruSnapshot {
	return gruSnapshot{
		Wz: n.Wz, Wr: n.Wr, Wh: n.Wh,
		Uz: n.Uz, Ur: n.Ur, Uh: n.Uh,
		Bz: n.Bz, Br: n.Br, Bh: n.Bh,
		Wq: n.Wq, Bq: n.Bq,
	}
}

func restoreGRU(n *Net, s gruSnapshot) {
	n.Wz, n.Wr, n.Wh = s.Wz, s.Wr, s.Wh
	n.Uz, n.Ur, n.Uh = s.Uz, s.Ur, s.Uh
	n.Bz, n.Br, n.Bh = s.Bz, s.Br, s.Bh
	n.Wq, n.Bq = s.Wq, s.Bq
}

// collectGRUGroups — по кілька представників кожної групи ваг разом з аналітичним
// градієнтом, знятим як (after − before)/lr.
//
// Саме ПО ГРУПАХ, а не суцільно: помилка живе у формулі конкретної групи (напр.
// reset-ворота), тож розбиття одразу вказує, де вона.
func collectGRUGroups(n *Net, before, after gruSnapshot, lr float32) []struct {
	name string
	live []*float32
	anal []float32
} {
	type grp = struct {
		name string
		live []*float32
		anal []float32
	}
	var out []grp

	// Крок по індексах: беремо представників з різних рядків/стовпців, а не сусідів.
	addMat := func(name string, live, b, a interface{}, rows, cols, step int) {
		g := grp{name: name}
		switch lv := live.(type) {
		case *[gruHidden][baseInputs]float32:
			bb := b.(*[gruHidden][baseInputs]float32)
			aa := a.(*[gruHidden][baseInputs]float32)
			for i := 0; i < rows; i += step {
				for j := 0; j < cols; j += step {
					g.live = append(g.live, &lv[i][j])
					g.anal = append(g.anal, (aa[i][j]-bb[i][j])/lr)
				}
			}
		case *[gruHidden][gruHidden]float32:
			bb := b.(*[gruHidden][gruHidden]float32)
			aa := a.(*[gruHidden][gruHidden]float32)
			for i := 0; i < rows; i += step {
				for j := 0; j < cols; j += step {
					g.live = append(g.live, &lv[i][j])
					g.anal = append(g.anal, (aa[i][j]-bb[i][j])/lr)
				}
			}
		case *[brainActions][gruHidden]float32:
			bb := b.(*[brainActions][gruHidden]float32)
			aa := a.(*[brainActions][gruHidden]float32)
			for i := 0; i < rows; i += step {
				for j := 0; j < cols; j += step {
					g.live = append(g.live, &lv[i][j])
					g.anal = append(g.anal, (aa[i][j]-bb[i][j])/lr)
				}
			}
		}
		out = append(out, g)
	}
	addVec := func(name string, live, b, a interface{}, size, step int) {
		g := grp{name: name}
		switch lv := live.(type) {
		case *[gruHidden]float32:
			bb := b.(*[gruHidden]float32)
			aa := a.(*[gruHidden]float32)
			for i := 0; i < size; i += step {
				g.live = append(g.live, &lv[i])
				g.anal = append(g.anal, (aa[i]-bb[i])/lr)
			}
		case *[brainActions]float32:
			bb := b.(*[brainActions]float32)
			aa := a.(*[brainActions]float32)
			for i := 0; i < size; i += step {
				g.live = append(g.live, &lv[i])
				g.anal = append(g.anal, (aa[i]-bb[i])/lr)
			}
		}
		out = append(out, g)
	}

	addMat("Wz (вхід→update)", &n.Wz, &before.Wz, &after.Wz, gruHidden, baseInputs, 5)
	addMat("Wr (вхід→reset)", &n.Wr, &before.Wr, &after.Wr, gruHidden, baseInputs, 5)
	addMat("Wh (вхід→кандидат)", &n.Wh, &before.Wh, &after.Wh, gruHidden, baseInputs, 5)
	addMat("Uz (h→update)", &n.Uz, &before.Uz, &after.Uz, gruHidden, gruHidden, 7)
	addMat("Ur (h→reset)", &n.Ur, &before.Ur, &after.Ur, gruHidden, gruHidden, 7)
	addMat("Uh (h→кандидат)", &n.Uh, &before.Uh, &after.Uh, gruHidden, gruHidden, 7)
	addVec("Bz (зсув update)", &n.Bz, &before.Bz, &after.Bz, gruHidden, 4)
	addVec("Br (зсув reset)", &n.Br, &before.Br, &after.Br, gruHidden, 4)
	addVec("Bh (зсув кандидата)", &n.Bh, &before.Bh, &after.Bh, gruHidden, 4)
	addMat("Wq (h→Q)", &n.Wq, &before.Wq, &after.Wq, brainActions, gruHidden, 3)
	addVec("Bq (зсув Q)", &n.Bq, &before.Bq, &after.Bq, brainActions, 1)
	return out
}

var _ = fmt.Sprintf
