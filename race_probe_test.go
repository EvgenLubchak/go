package main

import "testing"

// TestParallelPhaseHasNoRaceOnVelocity — паралельна фаза не пише швидкість.
//
// Інваріант фази: горутини пишуть ЛИШЕ свій AccX/AccY. Ухилення його порушувало —
// dodgeBurst писав VelX/VelY просто в паралельній фазі, а інша горутина в ту саму мить
// читала цю ж швидкість як швидкість СВОЄЇ цілі.
//
// ⚠️ УМОВА ВІДТВОРЕННЯ ВУЖЧА, НІЖ ЗДАЄТЬСЯ, і перша моя проба її не влучила.
// Хто читає ЧУЖУ швидкість у паралельній фазі — питання РОЗКЛАДКИ входів:
//
//	бойові БЕЗ поля (стражник, мисливець…)   → слоти 3-4 ВЛАСНІ, чужого не читають
//	рій (ConfigLearner)                      → слоти 3-4 = швидкість ЦІЛІ
//	вбивці (GatherKillerInputs)              → слоти 3-4 власні, АЛЕ 13-14 = ЦІЛІ
//
// (У коміті фікса стояло «GatherKillerInputs непричетний узагалі» — це ПОМИЛКА,
// слоти 3-4 сплутано зі слотом inTgtVelX = 13. Racy-читачами були і рій, і обидва
// вбивці; для проби беремо рій, бо його найпростіше поставити навпроти тих, хто
// ухиляється.) Із самих стражників детектор мовчав, і я мало не визнав ваду
// неіснуючою.
//
// Запускати з -race, інакше тест нічого не перевіряє:
//
//	go test -race -run TestParallelPhaseHasNoRaceOnVelocity
func TestParallelPhaseHasNoRaceOnVelocity(t *testing.T) {
	savedEps := qEpsilonConst
	qEpsilonConst = 0.9 // майже все випадкове → девʼята дія трапляється часто
	defer func() { qEpsilonConst = savedEps }()

	g := &Game{difficulty: 1.0, player: newPlayer()}
	g.player.X, g.player.Y = 5000, 5000 // подалі: цілями мусять бути юніти, не гравець

	add := func(cfg UnitConfig, faction, n int) {
		for i := 0; i < n; i++ {
			e := Pixel{
				X: 600 + float32(i%6)*26, Y: 600 + float32(i/6)*26 + float32(faction)*13,
				HP: 9, MaxHP: 9, Cfg: cfg, Faction: faction, Brain: NewBrain(),
			}
			e.Brain.combat, e.Brain.combatOnly = cfg.CombatReward, cfg.CombatOnly
			g.units = append(g.units, e)
		}
	}
	add(ConfigLearner, factionEnemy, 24)     // читають швидкість ЦІЛІ
	add(ConfigAllyChaser, factionPlayer, 24) // ухиляються

	for i := 0; i < 200; i++ {
		g.calcAcceleration()
		g.updateUnits() // саме тут застосовуються відкладені кидки
		for j := range g.units {
			g.units[j].DodgeCooldown, g.units[j].DodgeRecover = 0, 0 // не даємо гасити спроби
		}
	}
}
