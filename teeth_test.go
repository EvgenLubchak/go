package main

import (
	"math"
	"testing"
)

// TestTeethStayInsideTheMouth — зуби не вилазять за контур рота.
//
// Пастка тут неочевидна й мало не пройшла повз. Коли рот найбільш розкритий, капсула
// майже кругла, і РІВНОЇ кромки в неї майже немає: inner = w − 2r сходиться до нуля.
// При наших числах це 1.7 світових пікселя на всю ширину. Зуби, посаджені на пряму
// лінію, стирчали б з рота просто на обличчя — і читалось би це як дефект, а не як
// щелепа.
//
// Тому основа кожного зуба сідає на САМ контур капсули, і саме це тут перевіряється.
func TestTeethStayInsideTheMouth(t *testing.T) {
	for _, open := range []float32{0.4, 0.6, 0.8, 1.0} {
		w, h, r := mouthShape(open, 1)
		for i := 0; i < toothCount; i++ {
			tt := (float32(i) + 0.5) / float32(toothCount)
			dx := (tt - 0.5) * w
			half := mouthHalfHeightAt(dx, w, h, r)

			// Основа зуба мусить лежати НЕ ВИЩЕ кромки капсули в цій точці.
			if half > h/2+1e-4 {
				t.Errorf("розкриття %.1f, зуб %d: основа на %.2f вище кромки %.2f",
					open, i, half, h/2)
			}
			// І не нижче нуля: від'ємна піввисота означала б зуб за межами фігури.
			if half < 0 {
				t.Errorf("розкриття %.1f, зуб %d: піввисота %.2f", open, i, half)
			}
		}
	}

	// Контур мусить СПАДАТИ до країв — інакше це не капсула, і зуби сядуть не туди.
	w, h, r := mouthShape(1, 1)
	center := mouthHalfHeightAt(0, w, h, r)
	edge := mouthHalfHeightAt(w/2*0.98, w, h, r)
	if edge >= center {
		t.Errorf("кромка не звужується до краю: центр %.2f, край %.2f", center, edge)
	}
	if math.Abs(float64(mouthHalfHeightAt(w/2, w, h, r))) > 1e-4 {
		t.Error("на самому кінчику капсули висота мусить бути нульовою")
	}
}

// TestTeethAppearOnlyWhenMouthIsOpenEnough — у щілину зуби не лізуть.
//
// Без порога вони мигтіли б на кожному дрібному русі рота: при 6-8 пікселях на зуб
// половинчастий трикутник — це шум, а не деталь.
func TestTeethAppearOnlyWhenMouthIsOpenEnough(t *testing.T) {
	count := func(mouth float32) int {
		teethVerts, teethIdx = teethVerts[:0], teethIdx[:0]
		p := &Pixel{Mouth: mouth}
		w, h, r := mouthShape(mouth, 1)
		appendTeeth(p, 0, 0, w, h, r)
		return len(teethIdx) / 3
	}
	defer func() { teethVerts, teethIdx = teethVerts[:0], teethIdx[:0] }()

	if n := count(0); n != 0 {
		t.Errorf("закритий рот показав %d зубів", n)
	}
	if n := count(float32(toothMinOpen) - 0.01); n != 0 {
		t.Errorf("нижче порога показано %d зубів", n)
	}
	if n := count(1); n != toothCount*2 {
		t.Errorf("розкритий рот дав %d трикутників, очікували %d (по %d на щелепу)",
			n, toothCount*2, toothCount)
	}
}

// TestTeethBatchIntoOneCall — усі зуби кадру йдуть ОДНИМ викликом.
//
// Це і є причина, чому зуби взагалі дозволені собі коштувати: трикутника в пакеті
// vector немає, тож малюємо власним білим зображенням — а це інше джерело й розрив
// пакета. Розрив мусить бути ОДИН на кадр, а не на юніта; інакше ми повернемо ту саму
// ваду, яку виганяли з міток-шрифтів.
func TestTeethBatchIntoOneCall(t *testing.T) {
	teethVerts, teethIdx = teethVerts[:0], teethIdx[:0]
	defer func() { teethVerts, teethIdx = teethVerts[:0], teethIdx[:0] }()

	const units = 50
	w, h, r := mouthShape(1, 1)
	for i := 0; i < units; i++ {
		appendTeeth(&Pixel{Mouth: 1}, float32(i)*40, 0, w, h, r)
	}
	if got, want := len(teethIdx)/3, units*toothCount*2; got != want {
		t.Fatalf("накопичено %d трикутників, очікували %d", got, want)
	}
	// Індекси мусять бути наскрізними по всьому буферу: якби кожен юніт нумерувався
	// з нуля, одним викликом це намалювати було б неможливо.
	if int(teethIdx[len(teethIdx)-1]) != len(teethVerts)-1 {
		t.Error("індекси не наскрізні — зуби не складуться в один виклик")
	}
}
