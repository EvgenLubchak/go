package main

import "testing"

// TestHitstopFreezesOnlyThePlayer — заморозка спрацьовує на ударах ЗА УЧАСТЮ ГРАВЦЯ
// і не спрацьовує на бійці ботів між собою.
//
// Це не косметична деталь, а умова, без якої механіка не працює взагалі: при
// impactInvuln = 45 і півсотні юнітів бійня дає десятки влучань за секунду, і
// заморозка на кожному спинила б гру назавжди.
func TestHitstopFreezesOnlyThePlayer(t *testing.T) {
	newGame := func() *Game {
		g := &Game{difficulty: 1.0, player: newPlayer()}
		g.player.HP, g.player.MaxHP = 100, 100
		return g
	}

	// --- гравець ЗАВДАЄ удару: ривок в активній фазі, ворог упритул
	g := newGame()
	g.player.DashPhase = dashPhaseActive
	g.units = []Pixel{{
		X: g.player.X + pixelSize/2, Y: g.player.Y,
		HP: 10, MaxHP: 10, Faction: factionEnemy, Cfg: ConfigWarden,
	}}
	g.resolveImpacts()
	if g.hitstop != hitstopDealt {
		t.Errorf("завдав удару: заморозка %d, очікували %d", g.hitstop, hitstopDealt)
	}

	// --- гравець ОТРИМУЄ удару: ворог летить у нього досить швидко
	g = newGame()
	e := Pixel{
		X: g.player.X + pixelSize/2, Y: g.player.Y,
		HP: 10, MaxHP: 10, Faction: factionEnemy, Cfg: ConfigWarden,
	}
	// Напрямок від ворога до гравця — рівно вліво, тож уся швидкість іде в зближення.
	e.VelX = -impactThreshold(e.Cfg.MaxSpeed) * 2
	g.units = []Pixel{e}
	g.resolveImpacts()
	if g.hitstop != hitstopTaken {
		t.Errorf("отримав удару: заморозка %d, очікували %d", g.hitstop, hitstopTaken)
	}
	if hitstopTaken <= hitstopDealt {
		t.Errorf("асиметрія втрачена: отримав %d, завдав %d — отриманий удар мусить "+
			"триматись довше, бо гравцю треба встигнути зрозуміти, що сталось",
			hitstopTaken, hitstopDealt)
	}

	// --- БОТИ МІЖ СОБОЮ: жодної заморозки, хоч шкода й проходить
	g = newGame()
	g.player.X, g.player.Y = 10, 10 // подалі, щоб не втрапив у зіткнення
	a := Pixel{X: 1000, Y: 1000, HP: 10, MaxHP: 10, Faction: factionEnemy, Cfg: ConfigWarden}
	b := Pixel{X: 1000 + pixelSize/2, Y: 1000, HP: 10, MaxHP: 10, Faction: factionPlayer, Cfg: ConfigWarden}
	a.VelX = impactThreshold(a.Cfg.MaxSpeed) * 2
	g.units = []Pixel{a, b}
	g.resolveImpacts()
	if g.units[1].HP >= 10 {
		t.Fatal("бот не влучив у бота — перевіряти нічого, тест треба лагодити")
	}
	if g.hitstop != 0 {
		t.Errorf("бійка ботів заморозила світ на %d кадрів — при 50 юнітах це ступор",
			g.hitstop)
	}
}

// TestHitstopIgnoresBlockedHits — удар у невразливого або в того, хто ухилився, не
// морозить: це промах, а не подія.
func TestHitstopIgnoresBlockedHits(t *testing.T) {
	for _, tc := range []struct {
		name  string
		setup func(*Pixel)
	}{
		{"невразливість", func(p *Pixel) { p.InvulnTimer = 10 }},
		{"ухилення", func(p *Pixel) { p.DodgeTimer = 10 }},
	} {
		g := &Game{difficulty: 1.0, player: newPlayer()}
		g.player.HP, g.player.MaxHP = 100, 100
		g.player.DashPhase = dashPhaseActive
		e := Pixel{
			X: g.player.X + pixelSize/2, Y: g.player.Y,
			HP: 10, MaxHP: 10, Faction: factionEnemy, Cfg: ConfigWarden,
		}
		tc.setup(&e)
		g.units = []Pixel{e}
		g.resolveImpacts()
		if g.hitstop != 0 {
			t.Errorf("%s: заморозка %d — морозимо на подію, а не на спробу",
				tc.name, g.hitstop)
		}
	}
}

// TestHitstopCooldownStopsChaining — серія ударів не зчіплюється в суцільний ступор.
func TestHitstopCooldownStopsChaining(t *testing.T) {
	g := &Game{}
	g.freezeOnHit(hitstopTaken)
	if g.hitstop != hitstopTaken {
		t.Fatalf("перша заморозка не спрацювала: %d", g.hitstop)
	}
	g.hitstop = 0 // ніби кадри вже витекли
	g.freezeOnHit(hitstopTaken)
	if g.hitstop != 0 {
		t.Errorf("друга заморозка пройшла крізь кулдаун: %d кадрів", g.hitstop)
	}

	// Беремо БІЛЬШУ з двох, а не суму: в одному тіку можна і вдарити, і дістати.
	g = &Game{}
	g.freezeOnHit(hitstopDealt)
	g.hitstopCool = 0
	g.freezeOnHit(hitstopTaken)
	if g.hitstop != hitstopTaken {
		t.Errorf("заморозки склались або перетерлись: %d, очікували %d (більшу з двох)",
			g.hitstop, hitstopTaken)
	}
}

// TestHitstopDoesNotReachTheBench — головна властивість для ЗАМІРІВ.
//
// Усі записані базові лінії зняті на tickHeadless. Якби заморозка просочилась у
// симуляцію, вона тихо змінила б кількість тіків у вікні заміру — і всі порівняння з
// історією стали б недійсними, причому непомітно: числа лишились би правдоподібними.
//
// Тому шов проходить по Update, і цей тест його стереже.
func TestHitstopDoesNotReachTheBench(t *testing.T) {
	g := newBenchGame()
	g.hitstop = 999 // світ «заморожений» по максимуму
	before := g.tick
	drive := benchDriver(true, true)
	for i := 0; i < 10; i++ {
		g.tickHeadless(drive, true)
	}
	if got := g.tick - before; got != 10 {
		t.Errorf("стенд зробив %d тіків замість 10 — заморозка просочилась у заміри", got)
	}
	if g.hitstop != 999 {
		t.Errorf("стенд змінив лічильник заморозки (%d) — він мусить його ІГНОРУВАТИ, "+
			"а не споживати", g.hitstop)
	}
}
