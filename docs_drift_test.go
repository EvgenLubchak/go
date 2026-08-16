package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestDocsDoNotLieAboutConstants — числа в БЛОКАХ КОДУ документації збігаються з кодом.
//
// Проблему назвав сторонній огляд, і вона влучна: коментарі й доки — головний актив
// цього проєкту, а значить застарілий коментар це не дрібниця, а ЗІПСОВАНИЙ актив. За
// довідковою таблицею звіряються; якщо вона бреше, помиляється кожен, хто їй повірив.
//
// На момент написання розійшлись сім значень, серед них базові: brainActions (8 проти
// справжніх 9 після появи ухилення), усі три бойові нагороди (2.0/−2.0/5.0 проти
// 0.1/−0.1/0.25 після ділення на 20), playerMaxHP, whiskerRange, три константи
// феромонів. Оновити їх разово мало — вони розійдуться знову.
//
// [ПРАВИЛО, ЯКЕ ЦЕЙ ТЕСТ ВСТАНОВЛЮЄ]
//
//	ПРОЗА  може згадувати історичні числа — «було 2.0, поділили на 20». Це цінно, і
//	       забороняти це означало б викинути половину висновків;
//	БЛОК КОДУ (```go) — завжди ПОТОЧНИЙ стан. Він виглядає як лістинг, читається як
//	       лістинг, і брехати не має права.
//
// Тому перевіряємо лише блоки коду. Джерело істини — самі файли, а не список у тесті:
// інакше ми б просто перенесли дрейф на один рівень вище.
func TestDocsDoNotLieAboutConstants(t *testing.T) {
	assign := regexp.MustCompile(`(?m)^\s*([a-zA-Z][A-Za-z0-9_]{3,})\s*=\s*(?:float32\()?(-?\d+(?:\.\d+)?)\)?\s*(?://.*)?$`)

	// 1. Істина — з коду.
	truth := map[string]float64{}
	for _, f := range []string{"brain.go", "main.go", "tuning_world.go", "tuning_combat.go", "tuning_visual.go"} {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range assign.FindAllStringSubmatch(string(src), -1) {
			v, err := strconv.ParseFloat(m[2], 64)
			if err == nil {
				truth[m[1]] = v
			}
		}
	}
	if len(truth) < 50 {
		t.Fatalf("зібрано лише %d констант — розбір зламався, перевірка порожня", len(truth))
	}

	// 2. Претензії — з блоків коду в доках.
	docs, err := filepath.Glob("docs/*.md")
	if err != nil {
		t.Fatal(err)
	}
	inline := regexp.MustCompile(`([a-zA-Z][A-Za-z0-9_]{3,})\s*=\s*(-?\d+(?:\.\d+)?)`)
	bad := 0
	for _, d := range docs {
		src, err := os.ReadFile(d)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(src), "\n")
		inCode := false
		for i, line := range lines {
			if strings.HasPrefix(strings.TrimSpace(line), "```") {
				inCode = !inCode
				continue
			}
			if !inCode {
				continue // проза має право на історію
			}
			for _, m := range inline.FindAllStringSubmatch(line, -1) {
				want, ok := truth[m[1]]
				if !ok {
					continue // не наша ручка (поле структури, приклад, чужа назва)
				}
				got, err := strconv.ParseFloat(m[2], 64)
				if err != nil || got == want {
					continue
				}
				bad++
				t.Errorf("%s:%d  %s = %v у доках, а в коді %v\n\t%s",
					d, i+1, m[1], got, want, strings.TrimSpace(line))
			}
		}
	}
	if bad > 0 {
		t.Logf("розійшлось значень: %d. Проза може згадувати історію, блок коду — ні.", bad)
	}
}
