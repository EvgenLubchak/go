package main

import "testing"

// TestHeadOnDamagesBoth — третій рядок таблиці в шапці combat.go: «лоб-у-лоб на
// швидкості → обидва отримали (реальний ризик атаки)».
//
// ЧОМУ ЦЕ ВАРТО СТЕРЕГТИ, а не просто виправити раз. Рядок був недосяжний, і не
// через дрібну арифметику: applyImpactDamage відкидає ЦІЛЬ на knockbackImpulse
// (5.0), а перевірка другої сторони читала швидкість ПІСЛЯ цього. Щоб пережити
// імпульс і все одно вдарити, юнітові треба зближатись на поріг+5.0 ≈ 5.96 при
// стелі MaxSpeed 1.6 — тобто ніколи. Порядок двох рядків приховував цілий
// проєктний намір, і жоден тест його не покривав.
//
// Наслідок був не косметичний: із механіки зникав РИЗИК атаки, а саме з нього
// виводилось, що hit-and-run оптимальний математично. Без ризику таран у лоб
// безкарний для того, кого перевірили першим.
//
// [ПОРЯДОК У СЛАЙСІ — ГОЛОВНА ЧАСТИНА ТЕСТУ] Перевіряємо ОБА порядки й вимагаємо
// однакового результату. Це і є суть бага: цикл `for j := i+1` завжди перевіряв
// менший індекс першим, а індекс — лише позиція в unitRoster. Стражники стоять у
// ньому після юнітів гравця, тож глушився саме їхній удар — у типу, де нанесена
// шкода це ЄДИНЕ джерело позитивної нагороди (CombatOnly). Асиметрія від порядку
// створення тіл — це те, що тест мусить ловити, навіть якщо шкода в обидва боки
// колись стане несиметричною за задумом.
func TestHeadOnDamagesBoth(t *testing.T) {
	const hp = 10

	// Сцена: два ворожі один одному юніти впритул, летять НАЗУСТРІЧ по осі X.
	// Напрямок від першого до другого — рівно вправо, тож уся швидкість іде в
	// зближення й жодної тригонометрії в тесті немає.
	build := func(swap bool) *Game {
		g := &Game{difficulty: 1.0, player: newPlayer()}
		g.player.X, g.player.Y = 10, 10 // подалі: цикл гравця тут не предмет заміру

		left := Pixel{
			X: 1000, Y: 1000,
			HP: hp, MaxHP: hp, Faction: factionEnemy, Cfg: ConfigWarden,
		}
		right := Pixel{
			X: 1000 + pixelSize/2, Y: 1000,
			HP: hp, MaxHP: hp, Faction: factionPlayer, Cfg: ConfigWarden,
		}
		// Удвічі вище порога — щоб тест не стояв на межі й не падав від зміни
		// impactSpeedFrac. Знаки: лівий їде вправо, правий вліво.
		speed := impactThreshold(ConfigWarden.MaxSpeed) * 2
		left.VelX, right.VelX = speed, -speed

		if swap {
			g.units = []Pixel{right, left}
		} else {
			g.units = []Pixel{left, right}
		}
		return g
	}

	for _, tc := range []struct {
		name string
		swap bool
	}{
		{"лівий має менший індекс", false},
		{"правий має менший індекс", true},
	} {
		g := build(tc.swap)
		g.resolveImpacts()

		for i := range g.units {
			got := g.units[i].HP
			if got != hp-impactDamage {
				t.Errorf("%s: юніт[%d] HP %d, очікували %d — взаємна шкода не пройшла, "+
					"тобто лоб-у-лоб знову безкарний для того, кого перевірили першим",
					tc.name, i, got, hp-impactDamage)
			}
		}
	}
}

// TestPlayerDashTradeIsPossible — та сама пастка в циклі «гравець ↔ вороги»: твій
// ривок відкидав ворога ДО того, як міряли його швидкість, тож розмін «ти вдарив і
// дістав у відповідь» був недосяжний, а ривок не мав ціни.
//
// Стражника бʼють саме ривком, тож із цих кадрів він отримував лише біль — при тому
// що нанесена шкода його єдине джерело позитивної нагороди. Це найдорожчий випадок
// бага, і саме тому він тут окремим тестом, а не варіантом попереднього.
func TestPlayerDashTradeIsPossible(t *testing.T) {
	const enemyHP = 10

	g := &Game{difficulty: 1.0, player: newPlayer()}
	g.player.HP, g.player.MaxHP = 100, 100
	g.player.DashPhase = dashPhaseActive // гравець у активній фазі — ранить

	e := Pixel{
		X: g.player.X + pixelSize/2, Y: g.player.Y,
		HP: enemyHP, MaxHP: enemyHP, Faction: factionEnemy, Cfg: ConfigWarden,
	}
	// Напрямок від ворога до гравця — рівно вліво.
	e.VelX = -impactThreshold(ConfigWarden.MaxSpeed) * 2
	g.units = []Pixel{e}

	g.resolveImpacts()

	if g.units[0].HP != enemyHP-dashDamage {
		t.Errorf("ворог HP %d, очікували %d — ривок гравця не влучив, тест треба лагодити",
			g.units[0].HP, enemyHP-dashDamage)
	}
	if g.player.HP != 100-impactDamage {
		t.Errorf("гравець HP %d, очікували %d — ворог, що летів назустріч, не вкусив у "+
			"відповідь: ривок знову без ціни", g.player.HP, 100-impactDamage)
	}
}

// TestKnockbackStillOutrunsSpeed — контроль на сам тест вище.
//
// Обидва тести пройшли б і з неправильним фіксом, якби відкидання виявилось малим
// відносно швидкості: тоді взаємна шкода наставала б і за старим порядком, і тест
// нічого не стеріг би. Тому фіксуємо ту саму нерівність, через яку баг був тотальним:
// імпульс мусить лишатись більшим за все, чим юніт може його перебити. Якщо колись
// це перестане бути правдою — тести вище стануть порожніми, і повідомлення тут
// скаже, чому.
func TestKnockbackStillOutrunsSpeed(t *testing.T) {
	var fastest float32
	for _, cfg := range unitRoster {
		if cfg.MaxSpeed > fastest {
			fastest = cfg.MaxSpeed
		}
	}
	if fastest <= 0 {
		t.Fatal("у ростері немає жодної MaxSpeed — перевірка порожня")
	}
	if knockbackImpulse <= fastest {
		t.Logf("knockbackImpulse (%v) вже не перевищує найвищу MaxSpeed (%v): взаємна "+
			"шкода тепер настає й без порядку перевірок, тож TestHeadOnDamagesBoth "+
			"перестав стерегти саму пастку — перепиши його на явний порядок",
			float32(knockbackImpulse), fastest)
	}
}
