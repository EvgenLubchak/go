package main

import "testing"

// TestPreyDodgeExpires — таймер ухилення ЖЕРТВИ тікає, а не висить назавжди.
//
// Пастка суто структурна: гравець не входить у g.units, тож цикл таймерів в
// updateUnits його не бачить — та сама причина, через яку колись підвисав ворс. А
// updatePrey (режим aiPlayer) виставляє DodgeTimer при дії №8.
//
// Без тікання таймер лишався 20 назавжди, applyImpactDamage повертав false на кожен
// удар, і жертва ставала невразливою до кінця сесії. Рій фізично не міг її спіймати:
// спіймань немає → нагороди немає → «хижаки не навчились».
//
// Гра при цьому не падала й нічого не показувала — тому тест і потрібен.
func TestPreyDodgeExpires(t *testing.T) {
	g := &Game{difficulty: 1.0, player: newPlayer()}
	g.player.HP, g.player.MaxHP = 50, 50
	g.player.DodgeTimer = dodgeInvuln

	// Невразливість мусить СКІНЧИТИСЬ.
	for i := 0; i < dodgeInvuln+2; i++ {
		g.updatePlayer()
	}
	if g.player.DodgeTimer != 0 {
		t.Fatalf("невразливість жертви не скінчилась: %d кадрів лишилось", g.player.DodgeTimer)
	}

	// І тоді шкода мусить проходити. Це і є вся суть вади: рій не міг влучити.
	attacker := Pixel{X: g.player.X - 20, Y: g.player.Y}
	victim := g.player
	if !applyImpactDamage(&attacker, &victim, 1) {
		t.Error("після завершення ухилення жертва все ще невразлива")
	}

	// Стик із відходом — той самий, що в юнітів: інакше в жертви лишився б
	// БЕЗКОШТОВНИЙ дож, тобто рівно та домінантна дія, яку ми вилікували в стражників.
	if g.player.DodgeRecover == 0 {
		t.Error("після ухилення жертва не входить у відхід — дож знову безкоштовний")
	}
	for i := 0; i < dodgeRecovery+2; i++ {
		g.updatePlayer()
	}
	if g.player.DodgeRecover != 0 {
		t.Errorf("відхід жертви не скінчився: %d", g.player.DodgeRecover)
	}

	// І перезарядка теж мусить тікати, інакше друге ухилення не настане ніколи.
	g.player.DodgeCooldown = dodgeCooldown
	for i := 0; i < dodgeCooldown+2; i++ {
		g.updatePlayer()
	}
	if g.player.DodgeCooldown != 0 {
		t.Errorf("перезарядка жертви не скінчилась: %d", g.player.DodgeCooldown)
	}
}

// TestPreyDodgeResetsOnNewLife — новий епізод не починається з подарованої
// невразливості.
//
// respawnPlayer скидає HP, ворс, InvulnTimer і памʼять мозку — але ухилення довго
// лишалось поза цим списком. Жертва приходила б у нове життя з недотіклим таймером,
// і перші кадри епізоду були б безкарними.
func TestPreyDodgeResetsOnNewLife(t *testing.T) {
	g := &Game{difficulty: 1.0, player: newPlayer()}
	g.player.DodgeTimer, g.player.DodgeCooldown, g.player.DodgeRecover = 7, 8, 9
	g.respawnPlayer()
	if g.player.DodgeTimer != 0 || g.player.DodgeCooldown != 0 || g.player.DodgeRecover != 0 {
		t.Errorf("респаун лишив ухилення: timer=%d cooldown=%d recover=%d",
			g.player.DodgeTimer, g.player.DodgeCooldown, g.player.DodgeRecover)
	}
}
