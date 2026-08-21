package main

import (
	"math"
	"testing"
)

// TestQSpread — відрив Q₁−Q₂ рахується правильно на краях: найкраща дія перша,
// остання, відʼємні значення, і головне — ТОЧНИЙ нуль при нічиїй (саме він означає
// «argmax віддано шумові» і є підписом страху).
func TestQSpread(t *testing.T) {
	var q [brainActions]float32

	q = [brainActions]float32{5, 1, 0, 0, 0, 0, 0, 0, 0}
	if s := qSpread(q); s != 4 {
		t.Errorf("best на index 0: спред %v, очікувалось 4", s)
	}
	q = [brainActions]float32{0, 0, 0, 0, 0, 0, 0, 1, 5}
	if s := qSpread(q); s != 4 {
		t.Errorf("best на останньому index: спред %v, очікувалось 4", s)
	}
	q = [brainActions]float32{-3, -1, -2, -9, -9, -9, -9, -9, -9}
	if s := qSpread(q); s != 1 {
		t.Errorf("відʼємні Q: спред %v, очікувалось 1", s)
	}
	q = [brainActions]float32{2, 2, 0, 0, 0, 0, 0, 0, 0}
	if s := qSpread(q); s != 0 {
		t.Errorf("нічия двох найкращих: спред %v, очікувалось рівно 0", s)
	}
}

// TestFearCountersFlowToPanel — сантехніка: лічильники паніки з Brain доходять до
// вулика на панелі, порожній кадр НЕ тягне EMA до нуля, dodge накопичується як dmg
// і скидається клавішею M, а EMA-прилади скидання переживають.
func TestFearCountersFlowToPanel(t *testing.T) {
	g := &Game{}
	n := NewNet()
	n.file = "fear.json"
	b := NewBrainWith(n)
	b.combat = true
	b.mSpreadVisSum, b.mSpreadVisN = 1.2, 3 // середнє 0.4
	b.mSpreadBlindSum, b.mSpreadBlindN = 0.6, 2
	b.mFlipVisN, b.mDecVisN = 2, 4 // 50%
	b.mFlipBlindN, b.mDecBlindN = 1, 5
	b.mDodgeN, b.mDodgeTeleN = 7, 3
	g.units = []Pixel{{Brain: b}}

	g.metrics.collect(g)
	h := g.metrics.hives["fear.json"]
	if h == nil {
		t.Fatal("вулик не зареєструвався")
	}
	near := func(got, want float32) bool { return math.Abs(float64(got-want)) < 1e-5 }
	if !near(h.spreadVis, 0.4) || !near(h.spreadBlind, 0.3) {
		t.Errorf("спред: %v|%v, очікувалось 0.4|0.3", h.spreadVis, h.spreadBlind)
	}
	if !near(h.flipVis, 0.5) || !near(h.flipBlind, 0.2) {
		t.Errorf("flip: %v|%v, очікувалось 0.5|0.2", h.flipVis, h.flipBlind)
	}
	if h.dodgeN != 7 || h.dodgeTeleN != 3 {
		t.Errorf("dodge: %d/%d, очікувалось 7/3", h.dodgeN, h.dodgeTeleN)
	}
	if b.mSpreadVisN != 0 || b.mDodgeN != 0 || b.mDecBlindN != 0 {
		t.Error("лічильники Brain не скинулись після збору")
	}

	// Кадр БЕЗ рішень: EMA мусить тримати останнє значення, а не їхати до нуля
	// через порожній знаменник. Саме так виглядала б панель на паузі бою.
	g.metrics.collect(g)
	if !near(h.spreadVis, 0.4) || !near(h.flipVis, 0.5) {
		t.Errorf("порожній кадр зрушив EMA: dQ %v flip %v", h.spreadVis, h.flipVis)
	}

	// M: dodge (лічильник заміру) обнуляється, EMA-прилади («що зараз») — ні.
	g.metrics.resetCounters()
	if h.dodgeN != 0 || h.dodgeTeleN != 0 {
		t.Error("M не скинув лічильник dodge")
	}
	if !near(h.spreadVis, 0.4) {
		t.Error("M стер EMA-прилад — а він показує «зараз», не «скільки набігло»")
	}
}

// TestStepStackCountsFear — поведінковий: справжні рішення stepStack наповнюють
// лічильники правильно. Детермінізм — через frozenPolicy (ε=0) і зсув B3, який
// гарантує argmax незалежно від випадкової ініціалізації W3 (|Σ W3·h2| < 5).
func TestStepStackCountsFear(t *testing.T) {
	savedFrz := frozenPolicy
	frozenPolicy = true
	defer func() { frozenPolicy = savedFrz }()

	b := NewBrain()
	b.net.B3[actionDodge] = 100 // ухилення — беззаперечний argmax

	vis := [baseInputs]float32{}
	vis[inVisible] = 1
	vis[inDashAtMe] = 0.5 // замах на мене → натискання «tele»

	// 1: видячи, під замахом. Першому рішенню нема з чим порівнюватись → dec ще 0.
	if a := b.Step(vis, false); a != actionDodge {
		t.Fatalf("argmax не dodge (%d) — зсуву B3 замало", a)
	}
	// 2: видячи, замаху нема → dodge «сліпий» (спам).
	vis[inDashAtMe] = 0
	b.Step(vis, false)
	// 3: сліпий кадр — та сама дія, flip немає.
	var blind [baseInputs]float32
	b.Step(blind, false)
	// 4: перемикаємо фаворита на рух → перший справжній flip (на видячому кадрі).
	b.net.B3[actionDodge] = -100
	b.net.B3[2] = 100
	if a := b.Step(vis, false); a != 2 {
		t.Fatalf("argmax не перемкнувся на дію 2 (%d)", a)
	}

	if b.mDodgeN != 3 || b.mDodgeTeleN != 1 {
		t.Errorf("dodge %d/tele %d, очікувалось 3/1 (лише перше — під замахом)",
			b.mDodgeN, b.mDodgeTeleN)
	}
	if b.mDecVisN != 2 || b.mFlipVisN != 1 {
		t.Errorf("видячи: flip %d із %d рішень, очікувалось 1 із 2",
			b.mFlipVisN, b.mDecVisN)
	}
	if b.mDecBlindN != 1 || b.mFlipBlindN != 0 {
		t.Errorf("сліпо: flip %d із %d, очікувалось 0 із 1", b.mFlipBlindN, b.mDecBlindN)
	}
	if b.mSpreadVisN != 3 || b.mSpreadBlindN != 1 {
		t.Errorf("спред рахувався %d/%d рішень (vis/blind), очікувалось 3/1",
			b.mSpreadVisN, b.mSpreadBlindN)
	}
	if b.mSpreadVisSum <= 0 {
		t.Error("спред на домінантному argmax мусить бути додатним")
	}
}
