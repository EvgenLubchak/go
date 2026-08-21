package main

import "testing"

// TestDamageAttributesPerHive — шкода на панелі атрибутується ВУЛИКУ, а не котлу.
//
// Вада, яку це стереже, вже траплялась наживо: з двома бойовими вуликами на полі
// котловий рядок «dmg +17/−47» змішував завдане фіолетовими з отриманим стражниками,
// а різницю робили ривки гравця, які взагалі нічиї (у гравця немає мозку). Прочитати
// з такого рядка, ХТО виграє обміни, неможливо.
//
// Котел лишається навмисно — зі старими скрінами звіряються — але поруч мусить бути
// пер-вуликова правда.
func TestDamageAttributesPerHive(t *testing.T) {
	g := &Game{}
	nA, nB := NewNet(), NewNet()
	nA.file, nB.file = "hiveA.json", "hiveB.json"
	a, b := NewBrainWith(nA), NewBrainWith(nB)
	a.combat, b.combat = true, true
	a.mDmgDealt, a.mDmgTaken = 17, 5
	b.mDmgDealt, b.mDmgTaken = 3, 47
	// Третій — НЕ бойовий: його вулик не має показувати dmg узагалі.
	nC := NewNet()
	nC.file = "hiveC.json"
	c := NewBrainWith(nC)

	g.units = []Pixel{{Brain: a}, {Brain: b}, {Brain: c}}
	g.metrics.collect(g)

	hA, hB, hC := g.metrics.hives["hiveA.json"], g.metrics.hives["hiveB.json"], g.metrics.hives["hiveC.json"]
	if hA == nil || hB == nil || hC == nil {
		t.Fatal("вулики не зареєструвались у метриках")
	}
	if hA.dmgDealt != 17 || hA.dmgTaken != 5 {
		t.Errorf("вулик A: dmg +%d/-%d, очікувалось +17/-5", hA.dmgDealt, hA.dmgTaken)
	}
	if hB.dmgDealt != 3 || hB.dmgTaken != 47 {
		t.Errorf("вулик B: dmg +%d/-%d, очікувалось +3/-47", hB.dmgDealt, hB.dmgTaken)
	}
	if !hA.combat || !hB.combat || hC.combat {
		t.Errorf("прапорець combat: A=%v B=%v C=%v, очікувалось true/true/false",
			hA.combat, hB.combat, hC.combat)
	}
	// Котел — сума вуликів (порівнянність зі старими скрінами).
	if g.metrics.dmgDealt != 20 || g.metrics.dmgTaken != 52 {
		t.Errorf("котел: +%d/-%d, очікувалось +20/-52", g.metrics.dmgDealt, g.metrics.dmgTaken)
	}

	// Скидання заміру (клавіша M) обнуляє і пер-вуликову шкоду — це лічильник СВІТУ.
	g.metrics.resetCounters()
	if hA.dmgDealt != 0 || hA.dmgTaken != 0 || hB.dmgDealt != 0 || hB.dmgTaken != 0 {
		t.Error("resetCounters не обнулив пер-вуликову шкоду")
	}
}
