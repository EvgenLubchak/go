package main

import "testing"

// TestAimStyleCannotReachTheDodge — перевірка ЗДІЙСНЕННОСТІ перед заміром.
//
// Питання Фази 0 було: чи допоможе агентові знання стилю прицілу. Тоді, коли єдиним
// захистом був рух убік, воно мало сенс: щоб зійти з лінії удару, треба знати, куди
// та лінія піде.
//
// Зараз захист інший — девʼята дія дає НЕВРАЗЛИВІСТЬ, а напрямок кидка рахує
// dodgeBurst сам. Тобто агент обирає КОЛИ, але не КУДИ.
//
// Цей тест перевіряє ланцюжок із трьох ланок:
//
//  1. стилі СПРАВДІ різні — інакше моделювати нічого;
//  2. але на результат ухилення вони НЕ впливають — невразливість не питає, звідки
//     прилетіло;
//  3. і перевірка не порожня — без ухилення шкода проходить за обох стилів.
//
// Якщо ланка 2 тримається, оракул на теперішній механіці ЗОБОВʼЯЗАНИЙ дати нуль, і
// три години стенду підтвердять арифметику. Це рівно той випадок, заради якого ми й
// заводили перелік здійсненності: спершу «чи може агент виразити різницю», потім
// «чи вивчить».
func TestAimStyleCannotReachTheDodge(t *testing.T) {
	saved := dashStyleMirror
	defer func() { dashStyleMirror = saved }()

	attacker := Pixel{X: 100, Y: 100}
	// Ціль РУХАЄТЬСЯ — інакше ведучий і дзеркальний приціл збігаються за побудовою
	// (обидва дивляться в поточну позицію), і тест був би порожній.
	target := Pixel{X: 200, Y: 100, VelX: 0, VelY: 3, HP: 10, MaxHP: 10}

	// --- 1. Стилі дають РІЗНУ лінію удару
	dashStyleMirror = false
	leadX, leadY := dashAimAt(&attacker, &target)
	dashStyleMirror = true
	mirX, mirY := dashAimAt(&attacker, &target)
	if leadX == mirX && leadY == mirY {
		t.Fatal("стилі прицілу не відрізняються — моделювати нічого, тест порожній")
	}

	// --- 2. На результат УХИЛЕННЯ це не впливає
	for _, tc := range []struct {
		name   string
		mirror bool
	}{{"ведучий", false}, {"дзеркальний", true}} {
		dashStyleMirror = tc.mirror
		victim := target
		victim.DodgeTimer = dodgeInvuln // ухилення активне
		if applyImpactDamage(&attacker, &victim, 1) {
			t.Errorf("%s: ухилення не спрацювало — тест лагодити", tc.name)
		}
		if victim.HP != 10 {
			t.Errorf("%s: HP %d, ухилення мусить блокувати повністю", tc.name, victim.HP)
		}
	}

	// --- 3. Перевірка не порожня: без ухилення шкода проходить за обох стилів
	for _, tc := range []struct {
		name   string
		mirror bool
	}{{"ведучий", false}, {"дзеркальний", true}} {
		dashStyleMirror = tc.mirror
		victim := target
		if !applyImpactDamage(&attacker, &victim, 1) {
			t.Errorf("%s: без ухилення шкода мусить проходити — інакше ланка 2 нічого не доводить",
				tc.name)
		}
	}
}
