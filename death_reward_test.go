package main

import (
	"math"
	"testing"
)

// TestDeathRewardIgnoresVisibilityForDodge — термінальна нагорода після УХИЛЕННЯ не
// залежить від того, чи було видно ціль.
//
// Пастка суто арифметична: inWhisker0 = 5, actionDodge = 8, тож прямий індекс
// inWhisker0 + prevAction дає 13 — а це слот inVisible, «видно гравця». Штраф за рух у
// бік стіни (rewardNearWall = −0.3) приходив би за те, що ціль на видноті.
//
// Чому це важило більше, ніж здається: термінальні переходи рідкісні й найцінніші — у
// них увесь урок «смерть не безкоштовна». Саме вони й шуміли.
//
// Перевіряємо через РІЗНИЦЮ: два однакові переходи, які відрізняються лише
// видимістю. Абсолютне значення нагороди тут не важливе, важливо що воно НЕ ЗАЛЕЖИТЬ
// від слота, який до дії ухилення не має стосунку.
func TestDeathRewardIgnoresVisibilityForDodge(t *testing.T) {
	deathReward := func(action int, visible float32) float32 {
		g := &Game{}
		e := Pixel{HP: 0, Brain: NewBrain()}
		b := e.Brain
		b.hasPrev, b.prevAction = true, action
		b.prevState[inVisible] = visible
		g.deathTransition(&e)

		head := b.net.replayHead - 1
		if head < 0 {
			t.Fatal("перехід не записався")
		}
		return b.net.replay[head].r
	}

	seen, unseen := deathReward(actionDodge, 1), deathReward(actionDodge, 0)
	if d := math.Abs(float64(seen - unseen)); d > 1e-6 {
		t.Errorf("видимість цілі змінила термінальну нагороду за УХИЛЕННЯ на %.3f "+
			"(%.3f проти %.3f) — читається слот 13 замість вуса", d, seen, unseen)
	}

	// Перевірка не порожня: для РУХОВОЇ дії справжній вус на нагороду впливати мусить.
	// Без цієї половини тест проходив би й на «нагорода завжди стала».
	moved := func(w float32) float32 {
		g := &Game{}
		e := Pixel{HP: 0, Brain: NewBrain()}
		b := e.Brain
		b.hasPrev, b.prevAction = true, 2
		b.prevState[inWhisker0+2] = w
		g.deathTransition(&e)
		return b.net.replay[b.net.replayHead-1].r
	}
	if near, far := moved(1), moved(0); math.Abs(float64(near-far)) < 1e-6 {
		t.Error("справжній вус не впливає на нагороду — перевірка порожня, тест лагодити")
	}
}
