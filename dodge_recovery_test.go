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

// TestRecoveryBlocksControl — у відході юніт справді НЕ прискорюється.
//
// Додано після мутації: попередній тест дивився на лічильник, тож підміна умови
// блокування лишала його зеленим. Тобто він перевіряв, що відхід ТРИВАЄ, але не те,
// що відхід чогось КОШТУЄ, — а вся механіка саме в другому.
//
// Перевіряємо через справжній calcAcceleration: яку б дію не обрав мозок, у відході
// прискорення мусить лишитись нульовим. Усі вісім напрямків його дають, тож нуль тут
// однозначний.
func TestRecoveryBlocksControl(t *testing.T) {
	mk := func(recover int) *Game {
		g := &Game{difficulty: 1.0, player: newPlayer()}
		g.player.X, g.player.Y = 560, 500 // поруч, щоб ціль була видима й близька
		e := Pixel{X: 500, Y: 500, HP: 5, MaxHP: 5, Faction: factionEnemy, Cfg: ConfigWarden}
		e.Brain = NewBrain()
		e.Brain.combat, e.Brain.combatOnly = true, true
		e.DodgeRecover = recover
		g.units = []Pixel{e}
		return g
	}

	// У відході — тримаємо його весь час і дивимось, чи зрушить прискорення.
	g := mk(dodgeRecovery)
	for i := 0; i < 40; i++ {
		g.units[0].DodgeRecover = dodgeRecovery // не даємо витекти
		g.calcAcceleration()
		if g.units[0].AccX != 0 || g.units[0].AccY != 0 {
			t.Fatalf("у відході юніт прискорився: %.3f, %.3f — ціни немає",
				g.units[0].AccX, g.units[0].AccY)
		}
	}

	// Поза відходом — мусить рухатись хоч колись. Без цієї половини перевірка була б
	// порожньою: нульове прискорення «завжди» теж пройшло б перший цикл.
	g = mk(0)
	moved := false
	for i := 0; i < 40 && !moved; i++ {
		g.calcAcceleration()
		if g.units[0].AccX != 0 || g.units[0].AccY != 0 {
			moved = true
		}
	}
	if !moved {
		t.Error("поза відходом юніт теж не рухається — перевірка порожня, тест лагодити")
	}
}
