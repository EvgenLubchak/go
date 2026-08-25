package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// ==========================================================================
// [МЕТОДИКА] Важелі експерименту мусять бути ЯВНИМИ в harness-ах, а не успадкованими
// з глобалі.
//
// ПРОБЛЕМА, ЯКУ ЦЕ ЗАКРИВАЄ. doubleDQN живе в main.go як пакетна змінна, а стенд і
// драбина йдуть прямо в Net.tdUpdate — тобто беруть те значення, яке випадково стоїть
// у грі на момент прогону. Коли дефолт перевернули на true «на час експерименту», усі
// harness-и мовчки поїхали разом із ним. Записані в roadmap щаблі (34.4/62.3/69.7%
// утримання, γ0.95 нуль-з-шести проти γ0.99 шість-з-шести, нулі n-step/стратифікації/
// PER) знімались на КЛАСИЧНОМУ max, а будь-який повтор від сьогодні рахувався б із
// подвійною ціллю. Розбіжність не впала б в очі як зміна умов — вона виглядала б як
// флейкі, і найгірше, що на неї списали б реальний ефект.
//
// Це рівно той клас вади, від якого стереже docs_drift_test: не помилка в коді, а
// РОЗʼЇЗД між тим, що записано, і тим, що виконується.
// ==========================================================================

// recordedBaselineDDQN — значення doubleDQN, при якому зняті ЗАПИСАНІ бази.
//
// Іменована константа, а не літерал false у двох місцях: коли колись переміряємо
// драбину з подвійною ціллю, зміна мусить бути ОДНИМ свідомим рядком поруч із цим
// поясненням, а не двома правками в різних файлах.
//
// ⚠️ Міняти лише РАЗОМ із записом у roadmap: перезняті бази несумісні зі старими.
const recordedBaselineDDQN = false

// pinExperimentFlags фіксує важелі експерименту на значеннях запису й вертає їх назад.
//
// Через t.Cleanup, а не defer: helper викликається з ЧУЖОГО тіла, і defer тут
// спрацював би при виході з самого pinExperimentFlags, тобто негайно й безглуздо.
//
// ЧОМУ ЛИШЕ doubleDQN. Решта важелів (sharedBrain, localSight, memFrames…) harness-и
// вже виставляють ПОКОМІРКОВО й свідомо — це предмет заміру, а не тло. doubleDQN був
// єдиним, що протікав із гри. Додавати сюди щось «про всяк випадок» означало б тихо
// змінити умови прогонів, яких ми не робили.
func pinExperimentFlags(t *testing.T) {
	t.Helper()
	saved := doubleDQN
	doubleDQN = recordedBaselineDDQN
	t.Cleanup(func() { doubleDQN = saved })
}

// TestPinExperimentFlagsPinsAndRestores — сам пін працює: фіксує на час тесту й вертає.
//
// Тест на helper, а не на harness, і це не формальність: якби Cleanup не спрацьовував,
// прапорець лишався б false для ВСІХ наступних тестів у пакеті, і гра з ddqn:on мала б
// сюїту, що перевіряє ddqn:off. Мовчки.
func TestPinExperimentFlagsPinsAndRestores(t *testing.T) {
	before := doubleDQN
	t.Run("під піном", func(t *testing.T) {
		pinExperimentFlags(t)
		if doubleDQN != recordedBaselineDDQN {
			t.Errorf("doubleDQN = %v під піном, очікували %v", doubleDQN, recordedBaselineDDQN)
		}
	})
	if doubleDQN != before {
		t.Errorf("після підтесту doubleDQN = %v, а був %v — Cleanup не вернув прапорець, "+
			"і решта сюїти поїхала б за ним", doubleDQN, before)
	}
}

// TestHarnessesPinExperimentFlags — СТАТИЧНИЙ сторож: кожен вхід у стенд і драбину
// фіксує важелі.
//
// [ЧОМУ СТАТИЧНО, А НЕ ПРОГОНОМ] Перевірити це виконанням означало б запустити стенд
// або драбину — десятки хвилин, і саме тому їх сховано за прапорцями. Тож перевіряємо
// не поведінку, а СТРУКТУРУ: жодна точка входу не має права оминути пін. Це та сама
// ідея, що в docs_drift_test — читаємо власний вихідний код і вимагаємо від нього
// властивості.
//
// Розбір через go/ast, а не регуляркою: тіло функції треба обмежити точно, інакше
// виклик із сусідньої функції зарахувався б цій. Регулярка тут дала б сторожа, який
// бреше мовчки, — а такий гірший за відсутність сторожа.
//
// Головна цінність не в сьогоднішньому стані (він щойно виправлений руками), а в
// ЗАВТРАШНЬОМУ: наступна комірка стенду, дописана через півроку, впаде тут одразу,
// а не через місяць у вигляді «числа якісь не такі».
func TestHarnessesPinExperimentFlags(t *testing.T) {
	// ladderSkip — спільні ворота драбини; пін живе в ньому, тож виклик воріт
	// зараховуємо як пін. Що самі ворота пінять — перевіряємо окремо нижче.
	const gate = "ladderSkip"
	const pin = "pinExperimentFlags"

	fset := token.NewFileSet()
	checked := 0
	for _, file := range []string{"bench_test.go", "ladder_test.go"} {
		af, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatalf("%s: розбір зламався (%v) — перевірка порожня", file, err)
		}
		for _, d := range af.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || !strings.HasPrefix(fn.Name.Name, "Test") {
				continue
			}
			checked++
			if !callsAny(fn, pin, gate) {
				t.Errorf("%s: %s не фіксує важелі експерименту — прогін візьме doubleDQN "+
					"із гри, і числа розійдуться із записаними базами.\n"+
					"\tДодай %s(t) або %s(t) на початку.",
					file, fn.Name.Name, pin, gate)
			}
		}
	}
	if checked == 0 {
		t.Fatal("не знайдено жодної точки входу — розбір зламався, перевірка порожня")
	}
	t.Logf("перевірено точок входу: %d", checked)

	// Ворота драбини мусять пінити самі — інакше зарахування gate вище було б фікцією.
	af, err := parser.ParseFile(fset, "ladder_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var gateFn *ast.FuncDecl
	for _, d := range af.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == gate {
			gateFn = fn
		}
	}
	if gateFn == nil {
		t.Fatalf("%s не знайдено — сторож вище зараховував виклик, якого немає", gate)
	}
	if !callsAny(gateFn, pin) {
		t.Errorf("%s не викликає %s, хоча сторож вище зараховує його як пін — "+
			"тобто вся драбина лишилась би без фіксації", gate, pin)
	}
}

// TestCallsAnyDetectsAMissingPin — КОНТРОЛЬНИЙ СИГНАЛ на сторожа вище.
//
// Навіщо окремий тест. TestHarnessesPinExperimentFlags сьогодні зелений, і сам по собі
// це нічого не доводить: рівно так само він світився б зеленим, якби callsAny просто
// завжди повертав true. Сторож, який не вміє впасти, — не сторож, а прикраса, і
// найгірше в ньому те, що з ним перестаєш перевіряти руками.
//
// Перевіряємо детектор на СИНТЕТИЧНОМУ джерелі, а не мутацією справжніх harness-ів:
// правити відстежувані файли заради перевірки означало б лишити сюїту в стані, який
// залежить від того, чи відкотили ми правку. Тут же обидві відповіді детектора —
// «є» і «немає» — беруться з коду, який існує лише всередині цього тесту.
//
// Пастка, яку тест закриває заразом: тіло функції треба обмежувати ТОЧНО. Наївний
// пошук по всьому файлу зарахував би виклик із сусідньої функції, і withoutPin нижче
// «пройшов» би через withPin вище — саме тому розбір іде через go/ast, а не регуляркою.
func TestCallsAnyDetectsAMissingPin(t *testing.T) {
	const src = `package p

func withPin(t *testing.T) {
	pinExperimentFlags(t)
	doWork()
}

func withoutPin(t *testing.T) {
	doWork()
}
`
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "synthetic.go", src, 0)
	if err != nil {
		t.Fatalf("синтетичне джерело не розібралось: %v", err)
	}
	got := map[string]bool{}
	for _, d := range af.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok {
			got[fn.Name.Name] = callsAny(fn, "pinExperimentFlags")
		}
	}
	if len(got) != 2 {
		t.Fatalf("розібрано %d функцій замість 2 — перевірка порожня", len(got))
	}
	if !got["withPin"] {
		t.Error("детектор НЕ побачив наявного піну — сторож давав би хибну тривогу")
	}
	if got["withoutPin"] {
		t.Error("детектор ЗАРАХУВАВ функції чужий пін — сторож не впав би на пропуску, " +
			"тобто не стеріг би нічого")
	}
}

// callsAny — чи викликає тіло функції хоч одну з названих функцій.
func callsAny(fn *ast.FuncDecl, names ...string) bool {
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok {
			return true
		}
		for _, want := range names {
			if id.Name == want {
				found = true
				return false
			}
		}
		return true
	})
	return found
}
