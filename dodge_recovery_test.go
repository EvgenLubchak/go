package main

import "testing"

// TestDodgeRecoveryCostsSomething — у ухилення нарешті є ЦІНА, і вона у світі.
//
// Прилад-рот показав, що стражник тисне ухилення на кожному відкаті. Це був не збій
// навчання, а справжній оптимум: дія давала невразливість, пʼятикратний спринт (0.6 →
// 3.0) і швидкість, удвічі вищу за поріг удару, — і не коштувала нічого.
//
// Коментар у коді твердив, що ціна є: «втрачена швидкість зближення, тобто втрачений
// удар». Але цей член нагороди вимикається при CombatOnly, а саме він стоїть у
// стражника. Ціна описувала один тип, механікою користувався інший.
func TestDodgeRecoveryCostsSomething(t *testing.T) {
	g := &Game{difficulty: 1.0, player: newPlayer()}
	e := Pixel{X: 500, Y: 500, HP: 5, MaxHP: 5, Faction: factionEnemy, Cfg: ConfigWarden}
	e.DodgeTimer, e.DodgeRecover = 1, 0 // остання мить невразливості
	g.units = []Pixel{e}

	// --- СТИК: невразливість і безпорадність ідуть УПРИТУЛ.
	//
	// Найважливіше в усій механіці. Якби між ними лишився хоч один кадр, у якому юніт
	// уже вразливий, але ще керований, туди сховалась би стара домінантна політика:
	// кидок дав би всі переваги, а ціна пройшла б повз.
	g.updateUnits()
	if g.units[0].DodgeTimer != 0 {
		t.Fatalf("невразливість не скінчилась: %d", g.units[0].DodgeTimer)
	}
	if g.units[0].DodgeRecover != dodgeRecovery {
		t.Errorf("відхід не почався одразу після невразливості: %d, очікували %d",
			g.units[0].DodgeRecover, dodgeRecovery)
	}

	// --- у відході юніт УРАЗЛИВИЙ: інакше ціни знову не було б
	victim := g.units[0]
	attacker := Pixel{X: 480, Y: 500}
	if !applyImpactDamage(&attacker, &victim, 1) {
		t.Error("у відході шкода не проходить — ціна знову нульова")
	}

	// --- і НЕКЕРОВАНИЙ: відхід мусить справді коштувати керування
	g.units[0].AccX, g.units[0].AccY = 0, 0
	g.units[0].Brain = nil // не потрібен: перевіряємо саме гілку блокування
	before := g.units[0].DodgeRecover
	g.updateUnits()
	if g.units[0].DodgeRecover != before-1 {
		t.Errorf("лічильник відходу не тікає: %d → %d", before, g.units[0].DodgeRecover)
	}

	// --- відхід КІНЧАЄТЬСЯ: вічна безпорадність зробила б дію непридатною
	for i := 0; i < dodgeRecovery+5; i++ {
		g.updateUnits()
	}
	if g.units[0].DodgeRecover != 0 {
		t.Errorf("відхід не скінчився: %d", g.units[0].DodgeRecover)
	}
}

// TestMouthShowsRealHelplessness — рот показує ВІДХІД, а не перезарядку.
//
// Раніше він світився весь dodgeCooldown, і Євген слушно зауважив, що це слабка
// інформація: на перезарядці юніт цілком керований, просто не може ухилитись удруге.
// Роззявлений рот мусить означати «бий саме зараз», а це справедливо лише у відході.
func TestMouthShowsRealHelplessness(t *testing.T) {
	onCooldownOnly := &Pixel{DodgeCooldown: dodgeCooldown}
	if got := mouthTargetOf(onCooldownOnly); got != 0 {
		t.Errorf("сама перезарядка роззявила рот (%v) — це не безпорадність", got)
	}
	inRecovery := &Pixel{DodgeCooldown: dodgeCooldown, DodgeRecover: dodgeRecovery}
	if got := mouthTargetOf(inRecovery); got != 1 {
		t.Errorf("у відході рот не роззявився: %v", got)
	}
}

// TestRecoveryBlocksControl — у відході юніт справді НЕ кермує.
//
// Додано після мутації: попередній тест дивився на лічильник, тож підміна умови
// блокування лишала його зеленим. Тобто він перевіряв, що відхід ТРИВАЄ, але не що він
// чогось КОШТУЄ, — а вся механіка саме в другому.
func TestRecoveryBlocksControl(t *testing.T) {
	for _, tc := range []struct {
		name           string
		timer, recover int
		want           bool
	}{
		{"вільний", 0, 0, true},
		{"у кидку", dodgeInvuln, 0, false},
		{"у відході", 0, dodgeRecovery, false},
	} {
		p := &Pixel{DodgeTimer: tc.timer, DodgeRecover: tc.recover}
		if got := canSteer(p); got != tc.want {
			t.Errorf("%s: canSteer = %v, очікували %v", tc.name, got, tc.want)
		}
	}

	// Плюс наскрізна перевірка: хоч би що обрав мозок, у відході прискорення лишається
	// нульовим. Ця половина детермінована — усі вісім напрямків дають ненульове
	// прискорення, а девʼята дія теж не прискорює, тож нуль тут однозначний.
	g := &Game{difficulty: 1.0, player: newPlayer()}
	g.player.X, g.player.Y = 560, 500
	e := Pixel{X: 500, Y: 500, HP: 5, MaxHP: 5, Faction: factionEnemy, Cfg: ConfigWarden}
	e.Brain = NewBrain()
	e.Brain.combat, e.Brain.combatOnly = true, true
	g.units = []Pixel{e}
	for i := 0; i < 40; i++ {
		g.units[0].DodgeRecover = dodgeRecovery // тримаємо у відході
		g.calcAcceleration()
		if g.units[0].AccX != 0 || g.units[0].AccY != 0 {
			t.Fatalf("у відході юніт прискорився: %.3f, %.3f — ціни немає",
				g.units[0].AccX, g.units[0].AccY)
		}
	}
}

// TestDodgeCannotCancelItsOwnRecovery — з відходу не можна вистрибнути новим ухиленням.
//
// Пастка стає видимою, лише коли dodgeRecovery БІЛЬШИЙ за dodgeCooldown (у нас 100
// проти 60): перезарядка спливає, поки відхід ще триває. Без окремої умови юніт
// ухилявся б просто з власного відходу й скасовував щойно накладену ціну — механіка
// лишилась би на папері, а поведінка не змінилась би зовсім.
//
// Перевіряємо ПРАВИЛО прямо, а не через calcAcceleration: там дію обирає мозок, і при
// ε = 1% чекати від нього саме девʼятої дії означає писати плаваючий тест. Перша версія
// цього тесту так і зробила — і сама себе завалила на порожній половині.
func TestDodgeCannotCancelItsOwnRecovery(t *testing.T) {
	for _, tc := range []struct {
		name              string
		cooldown, recover int
		want              bool
	}{
		{"усе готове", 0, 0, true},
		{"відхід триває, перезарядка спливла", 0, dodgeRecovery / 2, false},
		{"перезарядка триває", dodgeCooldown / 2, 0, false},
		{"обидва тривають", dodgeCooldown / 2, dodgeRecovery / 2, false},
	} {
		p := &Pixel{DodgeCooldown: tc.cooldown, DodgeRecover: tc.recover}
		if got := canDodge(p); got != tc.want {
			t.Errorf("%s: canDodge = %v, очікували %v", tc.name, got, tc.want)
		}
	}

	// Другий випадок вище — і є вся суть. Він можливий ЛИШЕ тому, що відхід довший за
	// перезарядку; якби вони помінялись місцями, пастки б не існувало й тест нічого не
	// стеріг би. Тож стежимо і за самим співвідношенням.
	if dodgeRecovery <= dodgeCooldown {
		t.Logf("dodgeRecovery (%d) ≤ dodgeCooldown (%d): перезарядка й так довша, "+
			"тож умова DodgeRecover зараз надлишкова — але лишається як страховка",
			dodgeRecovery, dodgeCooldown)
	}
}
