package main

import (
	"math"
	"testing"
)

// TestDoubleDQNDecouplesSelectionFromEvaluation — подвійна ціль справді бере argmax
// із ЖИВОЇ мережі, а значення — з TARGET.
//
// Конструкція детермінована: W3 занулено, тож Q(a) == B3[a] точно. Живу й target
// навмисно розводимо: target вважає найкращою дію 3 (і оцінює її в 2), жива — дію 5
// (яку target оцінює в 0). Тоді всі чотири мислимі реалізації дають різні цілі,
// і обидві перевірки разом відрізняють правильну від будь-якої з трьох хибних:
//
//	обирає target, оцінює target (класика)      → γ·2   ← гілка прапорець OFF
//	обирає ЖИВА,  оцінює TARGET (подвійна)      → 0     ← гілка прапорець ON
//	обирає жива,  оцінює жива                   → γ·3      ловить гілка ON
//	обирає target, оцінює жива                  → 0        ловить гілка OFF (там γ·2)
//
// tdUpdate повертає сирий TD; при Q(s, a=0) = 0 він і є ціллю — звіряємо напряму.
func TestDoubleDQNDecouplesSelectionFromEvaluation(t *testing.T) {
	saved := doubleDQN
	defer func() { doubleDQN = saved }()

	build := func() *Net {
		n := NewNet()
		for a := range n.W3 {
			for k := range n.W3[a] {
				n.W3[a][k] = 0 // Q(a) == B3[a], без шуму випадкової ініціалізації
			}
		}
		n.B3[3] = 2 // зараз найкраща дія 3...
		n.syncTarget()
		n.B3[3] = 0
		n.B3[5] = 3 // ...а ЖИВА після «навчання» вважає найкращою дію 5
		return n
	}
	var s, s2 [brainInputs]float32

	doubleDQN = false
	n := build()
	td := n.tdUpdate(s, 0, 0, s2, false)
	want := n.gamma * 2 // target і обрала (дію 3), і оцінила (2)
	if math.Abs(float64(td-want)) > 1e-4 {
		t.Fatalf("класична ціль: TD %v, очікувалось %v (γ·Q_target[argmax_target])", td, want)
	}

	doubleDQN = true
	n = build()
	td = n.tdUpdate(s, 0, 0, s2, false)
	// Жива обрала дію 5, target чесно оцінив її в 0 → ціль 0. Якби оцінювала
	// жива — вийшло б γ·3; якби обирав target — γ·2. Обидва ловляться.
	if math.Abs(float64(td)) > 1e-4 {
		t.Fatalf("подвійна ціль: TD %v, очікувалось 0 (argmax живої, оцінка target)", td)
	}
}

// TestDoubleDQNDefaultIsOff — дефолт НЕ міняється до заміру (доктрина проєкту).
// Якщо експеримент виграв і дефолт свідомо перемикається — онови й цей тест,
// разом із записом результату в roadmap.
func TestDoubleDQNDefaultIsOff(t *testing.T) {
	if doubleDQN {
		t.Fatal("doubleDQN увімкнено дефолтом — дефолти не міняються до заміру")
	}
}
